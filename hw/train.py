from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyTokenizer:
    """Small deterministic tokenizer for Track A smoke-training."""

    def __init__(self, vocab_size: int = 512, image_token_id: int = 3) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.image_token_id = image_token_id
        self.image_start_token_id = 4
        self.image_end_token_id = 5

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0

        while i < len(text):
            if IMAGE_TOKEN and text.startswith(IMAGE_TOKEN, i):
                ids.append(self.image_token_id)
                i += len(IMAGE_TOKEN)
            elif IMAGE_START_TOKEN and text.startswith(IMAGE_START_TOKEN, i):
                ids.append(self.image_start_token_id)
                i += len(IMAGE_START_TOKEN)
            elif IMAGE_END_TOKEN and text.startswith(IMAGE_END_TOKEN, i):
                ids.append(self.image_end_token_id)
                i += len(IMAGE_END_TOKEN)
            else:
                ids.append(10 + (ord(text[i]) % (self.vocab_size - 10)))
                i += 1

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()

        return "A"


class TinyVisionEncoder(nn.Module):
    """Very small vision encoder for CPU smoke-training."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]
        pooled = pixel_values.mean(dim=(-1, -2))  # [B, 3]
        return self.proj(pooled)  # [B, hidden_size]


class TinyLanguageModel(nn.Module):
    """Tiny language model with inputs_embeds support.

    It is intentionally small and is used only to check that the training
    pipeline runs end-to-end on CPU.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            context = inputs_embeds.mean(dim=1, keepdim=True)
        else:
            mask = attention_mask.to(inputs_embeds.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            context = (inputs_embeds * mask).sum(dim=1, keepdim=True) / denom

        # Context makes answer-token logits depend on visual embeddings too,
        # so adapter receives gradients even when backbones are frozen.
        logits = self.lm_head(inputs_embeds + context)

        if labels is None:
            loss = logits.mean() * 0.0
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0]
        return torch.ones((batch_size, 1), dtype=torch.long, device=inputs_embeds.device)


def _make_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _make_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _ensure_image_token_positions(
    batch: dict[str, torch.Tensor],
    image_token_id: int,
    num_image_tokens: int,
    ignore_index: int,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Guarantee exactly num_image_tokens visual placeholders per sample.

    This keeps smoke-training robust even if special visual tokens are not
    represented by the tiny tokenizer exactly as in a real LLM tokenizer.
    """
    input_ids = batch["input_ids"].clone()
    labels = batch["labels"].clone()
    attention_mask = batch["attention_mask"].clone()

    if input_ids.shape[1] < num_image_tokens:
        raise ValueError("Sequence is shorter than num_image_tokens")

    input_ids[input_ids == image_token_id] = pad_token_id

    input_ids[:, :num_image_tokens] = image_token_id
    labels[:, :num_image_tokens] = ignore_index
    attention_mask[:, :num_image_tokens] = 1

    batch["input_ids"] = input_ids
    batch["labels"] = labels
    batch["attention_mask"] = attention_mask

    return batch


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(batch)

    if isinstance(output, dict):
        loss = output["loss"]
    elif hasattr(output, "loss"):
        loss = output.loss
    else:
        loss = output

    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss)

    if not torch.isfinite(loss).all():
        raise ValueError("Loss is not finite")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return float(loss.detach().cpu().item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    data_cfg = config.get("data", {})
    processor_cfg = config.get("processor", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    device = _make_device(str(trainer_cfg.get("device", "cpu")))
    dtype = _make_dtype(str(trainer_cfg.get("dtype", "float32")))

    manifest_path = data_cfg.get("train_manifest", "assets/toy_math_vqa/manifest.jsonl")
    split = data_cfg.get("split", "train")
    max_samples = data_cfg.get("max_samples")

    if fast_train:
        max_samples = min(int(max_samples or 8), 8)

    image_token_id = int(model_cfg.get("image_token_id", 3))
    vocab_size = int(model_cfg.get("vocab_size", 512))

    tokenizer = TinyTokenizer(vocab_size=vocab_size, image_token_id=image_token_id)

    processor_config = ProcessorConfig(
        image_size=int(processor_cfg.get("image_size", 224)),
        num_tiles=int(processor_cfg.get("num_tiles", 1)),
        tile_overlap=float(processor_cfg.get("tile_overlap", 0.0)),
        num_image_tokens=int(processor_cfg.get("num_image_tokens", 16)),
        max_length=int(processor_cfg.get("max_length", 256)),
        ignore_index=int(processor_cfg.get("ignore_index", -100)),
    )

    processor = MathVLMProcessor(tokenizer=tokenizer, config=processor_config)

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=split,
        max_samples=max_samples,
    )

    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(trainer_cfg.get("num_workers", 0)),
        collate_fn=lambda samples: processor.collate([processor(sample) for sample in samples]),
    )

    vision_hidden_size = int(model_cfg.get("vision_hidden_size", 32))
    text_hidden_size = int(model_cfg.get("text_hidden_size", 64))

    vision_encoder = TinyVisionEncoder(hidden_size=vision_hidden_size)
    language_model = TinyLanguageModel(vocab_size=vocab_size, hidden_size=text_hidden_size)

    model = MathVLM(
        vision_encoder=vision_encoder,
        language_model=language_model,
        config=ModelConfig(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            num_image_tokens=processor_config.num_image_tokens,
            image_token_id=image_token_id,
        ),
    )

    if bool(model_cfg.get("freeze_vision", True)) or bool(model_cfg.get("freeze_llm", True)):
        model.freeze_backbones()

    model.to(device=device)

    if dtype != torch.float32 and device.type != "cpu":
        model.to(dtype=dtype)

    trainable_params = [param for param in model.parameters() if param.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    max_steps = int(trainer_cfg.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 3)

    local_batch_size = batch_size
    global_batch_size = int(trainer_cfg.get("global_batch_size", local_batch_size))
    grad_accum_steps = max(1, math.ceil(global_batch_size / max(1, local_batch_size)))

    step = 0
    optimizer_step = 0
    losses: list[float] = []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in dataloader:
            batch = _ensure_image_token_positions(
                batch=batch,
                image_token_id=image_token_id,
                num_image_tokens=processor_config.num_image_tokens,
                ignore_index=processor_config.ignore_index,
                pad_token_id=tokenizer.pad_token_id,
            )
            batch = _move_batch_to_device(batch, device)

            output = model(batch)

            if isinstance(output, dict):
                loss = output["loss"]
            elif hasattr(output, "loss"):
                loss = output.loss
            else:
                loss = output

            if not torch.is_tensor(loss):
                loss = torch.as_tensor(loss)

            if not torch.isfinite(loss).all():
                raise ValueError("Loss is not finite")

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            step += 1
            losses.append(float(loss.detach().cpu().item()))

            if step % grad_accum_steps == 0 or step >= max_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            print(
                f"step={step} "
                f"optimizer_step={optimizer_step} "
                f"loss={losses[-1]:.6f}"
            )

            if step >= max_steps:
                break

        if step >= max_steps:
            break

    save_path = trainer_cfg.get("save_checkpoint_path")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "adapter_state_dict": model.adapter.state_dict(),
                "config": config,
                "losses": losses,
            },
            save_path,
        )

        print(f"Saved checkpoint to {save_path}")

    if losses:
        print(f"final_loss={losses[-1]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()