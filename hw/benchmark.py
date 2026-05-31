from __future__ import annotations

import argparse
import json
import math
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


def _option_numeric_values(options: list[str]) -> dict[str, float]:
    """Extract numeric values from options like 'A) 13' or 'C) 135°'."""
    values: dict[str, float] = {}

    for option in options:
        match = re.match(r"\s*([A-Z])\)\s*(-?\d+(?:[.,]\d+)?)", option.strip(), re.IGNORECASE)
        if not match:
            continue

        letter = match.group(1).upper()
        value = float(match.group(2).replace(",", "."))
        values[letter] = value

    return values


def _choose_numeric_answer(target: float, options: list[str], tol: float = 1e-6) -> str | None:
    values = _option_numeric_values(options)

    for letter, value in values.items():
        if abs(value - target) <= tol:
            return letter

    if values:
        return min(values, key=lambda letter: abs(values[letter] - target))

    return None


def _solve_from_text(question: str, options: list[str]) -> str | None:
    """Small deterministic solver for toy visual-math questions.

    It does not use gold labels. It only parses the question and answer options.
    This is a simple Track A baseline, not a real VLM.
    """
    text = question.lower()

    # Right triangle: "катеты 5 и 12", "катетами 3 и 4"
    if "гипотенуз" in text:
        numbers = [int(x) for x in re.findall(r"\d+", text)]
        if len(numbers) >= 2:
            a, b = numbers[0], numbers[1]
            hypotenuse = math.sqrt(a * a + b * b)
            return _choose_numeric_answer(hypotenuse, options)

    # Rectangle area: "ширина 5 и высота 2"
    if "площад" in text and "ширин" in text and "высот" in text:
        numbers = [int(x) for x in re.findall(r"\d+", text)]
        if len(numbers) >= 2:
            width, height = numbers[0], numbers[1]
            return _choose_numeric_answer(width * height, options)

    # Supplementary angle: "смежный", angle = 180 - given angle
    if "смежн" in text and "угол" in text:
        numbers = [int(x) for x in re.findall(r"\d+", text)]
        if numbers:
            angle = numbers[0]
            return _choose_numeric_answer(180 - angle, options)

    # Line equation examples: y=2x+1, x=2
    if "y" in text and "x" in text:
        eq_match = re.search(r"y\s*=\s*([+-]?\d*)\s*x\s*([+-]\s*\d+)?", text)
        x_match = re.search(r"x\s*=\s*([+-]?\d+)", text)

        if eq_match and x_match:
            coef_text = eq_match.group(1)
            if coef_text in ("", "+"):
                a = 1
            elif coef_text == "-":
                a = -1
            else:
                a = int(coef_text)

            b_text = eq_match.group(2)
            b = int(b_text.replace(" ", "")) if b_text else 0
            x = int(x_match.group(1))

            return _choose_numeric_answer(a * x + b, options)

    return None


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


def _baseline_generate(question: str, options: list[str]) -> str:
    """Deterministic Track A baseline.

    First tries to solve simple symbolic math from text.
    Falls back to A when the answer requires visual parsing.
    """
    solved = _solve_from_text(question, options)
    if solved is not None:
        return solved

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

        raw_output = _baseline_generate(sample.question, sample.options)
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