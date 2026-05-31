from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample."""

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        image = image.convert("RGB")
        image_size = self.config.image_size
        num_tiles = self.config.num_tiles

        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC

        def image_to_tensor(img: Image.Image) -> torch.Tensor:
            img = img.resize((image_size, image_size), resample)
            tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
            tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1)
            return tensor / 255.0

        if num_tiles <= 1:
            return image_to_tensor(image).unsqueeze(0)

        width, height = image.size

        cols = 1
        while cols * cols < num_tiles:
            cols += 1
        rows = (num_tiles + cols - 1) // cols

        tile_tensors: list[torch.Tensor] = []

        for row_idx in range(rows):
            for col_idx in range(cols):
                if len(tile_tensors) >= num_tiles:
                    break

                left = col_idx * width // cols
                upper = row_idx * height // rows
                right = (col_idx + 1) * width // cols
                lower = (row_idx + 1) * height // rows

                tile = image.crop((left, upper, right, lower))
                tile_tensors.append(image_to_tensor(tile))

        return torch.stack(tile_tensors, dim=0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options."""
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        visual_part = f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}".strip()

        options_text = "\n".join(sample.options)

        prompt = (
            f"{visual_part}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            f"Ответ:"
        )

        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample."""
        prompt = self.build_prompt(sample, include_answer=False)
        answer = f" {sample.answer}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=True)

        input_ids = prompt_ids + answer_ids
        labels = [self.config.ignore_index] * len(prompt_ids) + answer_ids

        input_ids = input_ids[: self.config.max_length]
        labels = labels[: self.config.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values."""
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        max_len = max(item["input_ids"].shape[0] for item in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.config.ignore_index, dtype=torch.long)

        for idx, item in enumerate(batch):
            length = item["input_ids"].shape[0]

            input_ids[idx, :length] = item["input_ids"]
            attention_mask[idx, :length] = item["attention_mask"]
            labels[idx, :length] = item["labels"]

        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }