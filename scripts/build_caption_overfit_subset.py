#!/usr/bin/env python3
"""Select a deterministic 100-caption ELF overfitting dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic caption subset for ELF overfitting."
    )
    parser.add_argument(
        "--input-dataset",
        type=Path,
        default=Path(
            "outputs/processed/audiocaps_caption_only/hf_dataset/train"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/elf_caption_overfit_100/data"),
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        "--n-workers",
        type=int,
        default=0,
        help="Arrow save processes; 0 selects up to 16 available CPUs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def available_workers() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return min(16, max(1, count))


def prepare_output(path: Path, overwrite: bool) -> None:
    absolute = Path(os.path.abspath(path))
    if absolute in {Path("/"), Path.home(), Path.cwd().absolute()}:
        raise ValueError(f"Refusing broad output directory: {absolute}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}; use --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_dataset(dataset) -> None:
    required = {"input_ids", "sequence_length", "caption", "youtube_id"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    if len(dataset) == 0:
        raise ValueError("Input dataset is empty")


def select_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if dataset_size < num_samples:
        raise ValueError(
            f"Requested {num_samples} examples, but dataset contains {dataset_size}"
        )
    generator = random.Random(seed)
    return sorted(generator.sample(range(dataset_size), num_samples))


def write_references(dataset, path: Path, tqdm_class) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in tqdm_class(dataset, desc="Write overfit references", unit="captions"):
            record: dict[str, Any] = {
                "index": int(row["index"]) if row.get("index") is not None else None,
                "caption": row["caption"],
                "youtube_id": row["youtube_id"],
                "audio_id": row.get("audio_id"),
                "audio_path": row.get("audio_path"),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_from_disk
        from tqdm.auto import tqdm
    except ImportError as exc:
        print(
            "ERROR: install dependencies with `pip install -r requirements-data.txt`",
            file=sys.stderr,
        )
        return 1

    workers = args.num_workers or available_workers()
    try:
        dataset = load_from_disk(str(args.input_dataset))
        validate_dataset(dataset)
        indices = select_indices(len(dataset), args.num_samples, args.seed)
        prepare_output(args.output_dir, args.overwrite)
        selected_rows = [
            dataset[index]
            for index in tqdm(indices, desc="Select overfit captions", unit="captions")
        ]
        selected = dataset.select(indices)

        effective_workers = min(workers, len(selected))
        train_path = args.output_dir / "train"
        selected.save_to_disk(str(train_path), num_proc=effective_workers)
        write_references(selected_rows, args.output_dir / "references.jsonl", tqdm)

        lengths = [int(row["sequence_length"]) for row in selected_rows]
        unique_captions = len({str(row["caption"]).strip() for row in selected_rows})
        summary = {
            "input_dataset": str(args.input_dataset),
            "output_dataset": str(train_path),
            "references": str(args.output_dir / "references.jsonl"),
            "seed": args.seed,
            "requested_samples": args.num_samples,
            "selected_samples": len(selected),
            "unique_captions": unique_captions,
            "num_workers": effective_workers,
            "selected_source_indices": indices,
            "token_length": {
                "min": min(lengths),
                "mean": sum(lengths) / len(lengths),
                "max": max(lengths),
            },
            "passed": len(selected) == args.num_samples,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Selected {len(selected)} captions with seed {args.seed}")
    print(f"ELF data_path: {train_path}")
    print(f"References: {args.output_dir / 'references.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
