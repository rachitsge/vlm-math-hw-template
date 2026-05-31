from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        if vision_hidden_states.ndim == 2:
            vision_hidden_states = vision_hidden_states.unsqueeze(1)

        if vision_hidden_states.ndim != 3:
            raise ValueError("vision_hidden_states must have shape [B, N, D] or [B, D]")

        if vision_hidden_states.shape[1] != self.num_image_tokens:
            vision_hidden_states = torch.nn.functional.adaptive_avg_pool1d(
                vision_hidden_states.transpose(1, 2),
                self.num_image_tokens,
            ).transpose(1, 2)

        return self.proj(vision_hidden_states)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings."""
    merged = input_embeds.clone()

    batch_size = input_ids.shape[0]

    for batch_idx in range(batch_size):
        image_positions = (input_ids[batch_idx] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        num_positions = image_positions.numel()

        if num_positions != visual_embeds.shape[1]:
            raise ValueError(
                f"Expected {visual_embeds.shape[1]} image token positions, got {num_positions}"
            )

        merged[batch_idx, image_positions] = visual_embeds[batch_idx]

    return merged


def _extract_hidden_states(output: Any) -> torch.Tensor:
    """Extract tensor hidden states from common model output formats."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("last_hidden_state", "pooler_output", "hidden_states", "logits"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]

    for attr in ("last_hidden_state", "pooler_output", "hidden_states", "logits"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if torch.is_tensor(value):
                return value
            if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[-1]):
                return value[-1]

    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first

    raise ValueError("Cannot extract hidden states from vision encoder output")


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images and return [B, N, vision_hidden_size]."""
        if pixel_values.ndim == 5:
            batch_size, num_tiles = pixel_values.shape[:2]
            flat_pixels = pixel_values.view(
                batch_size * num_tiles,
                *pixel_values.shape[2:],
            )
            vision_output = self.vision_encoder(flat_pixels)
            hidden_states = _extract_hidden_states(vision_output)

            if hidden_states.ndim == 3:
                hidden_states = hidden_states.mean(dim=1)

            hidden_states = hidden_states.view(batch_size, num_tiles, -1)
            return hidden_states

        vision_output = self.vision_encoder(pixel_values)
        hidden_states = _extract_hidden_states(vision_output)

        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)

        return hidden_states

    def _build_inputs_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        vision_hidden_states = self._encode_images(pixel_values)
        visual_embeds = self.adapter(vision_hidden_states)

        embedding_layer = self.language_model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids)

        return merge_visual_embeddings(
            input_embeds=input_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        inputs_embeds = self._build_inputs_embeds(batch)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        inputs_embeds = self._build_inputs_embeds(batch)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )