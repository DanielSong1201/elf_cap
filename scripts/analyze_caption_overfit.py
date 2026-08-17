#!/usr/bin/env python3
"""Analyze memorization and diversity in ELF overfit generations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


GENERATED_PATTERN = re.compile(r"all_generated_(\d+)_(\d+)\.jsonl$")
LOSS_PATTERN = re.compile(
    r"Step\s+(\d+):\s+loss=([0-9.eE+-]+),\s+"
    r"l2=([0-9.eE+-]+),\s+ce=([0-9.eE+-]+)"
)
WORD_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ELF caption overfit generations against 100 references."
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(
            "outputs/experiments/elf_caption_overfit_100/data/references.jsonl"
        ),
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=Path("outputs/experiments/elf_caption_overfit_100/train"),
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path("outputs/experiments/elf_caption_overfit_100/train.log"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/elf_caption_overfit_100/analysis"),
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.lower()))


def load_jsonl(path: Path, field: str, tqdm_class) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in tqdm_class(stream, desc=f"Read {path.name}", unit="lines"):
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get(field)
            if isinstance(value, str):
                values.append(value.strip())
    return values


def repeated_bigram(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    return len(bigrams) != len(set(bigrams))


def nearest_reference(text: str, references: list[str]) -> tuple[int, float]:
    normalized = normalize_text(text)
    best_index = 0
    best_score = -1.0
    for index, reference in enumerate(references):
        score = SequenceMatcher(
            None, normalized, normalize_text(reference), autojunk=False
        ).ratio()
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def analyze_generation(
    path: Path, references: list[str], max_examples: int, tqdm_class
) -> dict[str, Any]:
    generated = load_jsonl(path, "generated", tqdm_class)
    normalized_refs = [normalize_text(text) for text in references]
    reference_set = set(normalized_refs)
    normalized_generated = [normalize_text(text) for text in generated]
    nonempty = [text for text in generated if normalize_text(text)]
    nonempty_normalized = [normalize_text(text) for text in nonempty]

    nearest: list[dict[str, Any]] = []
    for text in tqdm_class(nonempty, desc=f"Match {path.name}", unit="samples"):
        reference_index, similarity = nearest_reference(text, references)
        nearest.append(
            {
                "generated": text,
                "nearest_reference": references[reference_index],
                "similarity": similarity,
                "exact_match": normalize_text(text) in reference_set,
            }
        )

    exact_values = {text for text in nonempty_normalized if text in reference_set}
    word_lists = [WORD_PATTERN.findall(text.lower()) for text in nonempty]
    match = GENERATED_PATTERN.search(path.name)
    epoch = int(match.group(1)) if match else None
    step = int(match.group(2)) if match else None
    similarities = [item["similarity"] for item in nearest]

    closest = sorted(nearest, key=lambda item: item["similarity"], reverse=True)
    return {
        "generation_file": str(path),
        "epoch": epoch,
        "step": step,
        "num_generated": len(generated),
        "num_nonempty": len(nonempty),
        "nonempty_rate": len(nonempty) / max(1, len(generated)),
        "unique_nonempty": len(set(nonempty_normalized)),
        "unique_ratio": len(set(nonempty_normalized)) / max(1, len(nonempty)),
        "exact_match_samples": sum(text in reference_set for text in nonempty_normalized),
        "exact_match_rate": (
            sum(text in reference_set for text in nonempty_normalized)
            / max(1, len(nonempty))
        ),
        "training_caption_coverage": len(exact_values) / max(1, len(reference_set)),
        "mean_nearest_similarity": statistics.fmean(similarities) if similarities else 0.0,
        "max_nearest_similarity": max(similarities) if similarities else 0.0,
        "mean_words": (
            statistics.fmean(len(words) for words in word_lists) if word_lists else 0.0
        ),
        "repeated_bigram_rate": (
            sum(repeated_bigram(words) for words in word_lists) / max(1, len(word_lists))
        ),
        "most_common_generations": Counter(nonempty_normalized).most_common(max_examples),
        "closest_examples": closest[:max_examples],
    }


def parse_training_log(path: Path, tqdm_class) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    points: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in tqdm_class(stream, desc="Parse training loss", unit="lines"):
            match = LOSS_PATTERN.search(line)
            if match:
                points.append(
                    {
                        "step": int(match.group(1)),
                        "loss": float(match.group(2)),
                        "l2_loss": float(match.group(3)),
                        "ce_loss": float(match.group(4)),
                    }
                )
    if not points:
        return {"log": str(path), "points": 0}
    return {
        "log": str(path),
        "points": len(points),
        "first": points[0],
        "last": points[-1],
        "minimum_loss": min(points, key=lambda item: item["loss"]),
        "series": points,
    }


def write_metrics_csv(path: Path, analyses: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "step",
        "num_generated",
        "nonempty_rate",
        "unique_ratio",
        "exact_match_rate",
        "training_caption_coverage",
        "mean_nearest_similarity",
        "max_nearest_similarity",
        "mean_words",
        "repeated_bigram_rate",
        "generation_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for analysis in analyses:
            writer.writerow({field: analysis[field] for field in fields})


def write_loss_csv(path: Path, training_loss: dict[str, Any]) -> None:
    series = training_loss.get("series", [])
    if not series:
        return
    fields = ["step", "loss", "l2_loss", "ce_loss"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(series)


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        print("ERROR: --max-examples must be >= 0", file=sys.stderr)
        return 2
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("ERROR: install tqdm with `pip install -r requirements-data.txt`", file=sys.stderr)
        return 1

    try:
        references = load_jsonl(args.references, "caption", tqdm)
        if not references:
            raise ValueError(f"No captions found in {args.references}")
        generation_files = sorted(
            args.generation_root.rglob("all_generated_*.jsonl"),
            key=lambda path: (
                (GENERATED_PATTERN.search(path.name).groups()
                 if GENERATED_PATTERN.search(path.name) else ("0", "0"))
            ),
        )
        if not generation_files:
            raise FileNotFoundError(
                f"No all_generated_*.jsonl files under {args.generation_root}"
            )
        analyses = [
            analyze_generation(path, references, args.max_examples, tqdm)
            for path in tqdm(generation_files, desc="Analyze checkpoints", unit="files")
        ]
        analyses.sort(key=lambda item: (item["epoch"] or 0, item["step"] or 0))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_loss = parse_training_log(args.train_log, tqdm)
        report = {
            "references": str(args.references),
            "num_references": len(references),
            "generation_root": str(args.generation_root),
            "training_loss": training_loss,
            "checkpoints": analyses,
            "latest": analyses[-1],
        }
        (args.output_dir / "analysis.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_metrics_csv(args.output_dir / "metrics.csv", analyses)
        if training_loss is not None:
            write_loss_csv(args.output_dir / "loss_curve.csv", training_loss)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    latest = analyses[-1]
    print(f"Analyzed {len(analyses)} generation checkpoints")
    print(f"Latest exact-match rate: {latest['exact_match_rate']:.3f}")
    print(f"Latest nearest similarity: {latest['mean_nearest_similarity']:.3f}")
    print(f"Latest unique ratio: {latest['unique_ratio']:.3f}")
    print(f"Report: {args.output_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
