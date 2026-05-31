from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES
from hw.dataset import MathVQADataset


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    if text is None:
        return None

    allowed = set(choices)
    text = str(text).strip().upper()

    patterns = [
        r"^\s*\(?([A-Z])\)?\s*[\.\):]?\s*$",
        r"(?:ANSWER|ОТВЕТ|CORRECT ANSWER|THE CORRECT ANSWER IS)\s*[:\-]?\s*\(?([A-Z])\)?",
        r"\b([A-Z])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer in allowed:
                return answer

    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)

    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})

    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))

    return metrics


def _baseline_generate(prompt: str, options: list[str]) -> str:
    """Tiny deterministic baseline for Track A toy benchmark.

    This is not a trained VLM. It only allows the benchmark pipeline to run
    end-to-end on CPU.
    """
    return "A"


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    data_cfg = config.get("data", {})
    inference_cfg = config.get("inference", {})

    manifest_path = data_cfg.get("eval_manifest", "assets/toy_math_vqa/manifest.jsonl")
    split = data_cfg.get("split", "dev")
    max_samples = data_cfg.get("max_samples")

    if toy:
        manifest_path = "assets/toy_math_vqa/manifest.jsonl"
        split = "dev"

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=split,
        max_samples=max_samples,
    )

    rows: list[dict[str, Any]] = []

    for sample in dataset:
        prompt = build_benchmark_prompt(sample.question, sample.options)

        raw_output = _baseline_generate(prompt, sample.options)
        prediction = parse_mc_answer(raw_output)

        if prediction is None:
            prediction = normalize_text(raw_output)

        answer = sample.answer.strip()
        if answer.upper() in CHOICES:
            answer = answer.upper()
        else:
            answer = normalize_text(answer)

        rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "prompt": prompt,
                "raw_output": raw_output,
                "prediction": prediction,
                "answer": answer,
                "correct": prediction == answer,
            }
        )

    output_path = inference_cfg.get("output_path")
    if output_path:
        _write_jsonl(output_path, rows)

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()