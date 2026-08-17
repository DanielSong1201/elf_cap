#!/usr/bin/env python3
"""Prepare AudioCaps captions for ELF caption-only domain adaptation.

Input manifests contain one record per audio with a ``captions`` list. This
script expands them to one text example per valid caption, writes auditable
JSONL files, tokenizes captions with the ELF T5 tokenizer, and saves one
Hugging Face Arrow dataset per split for direct use by ELF training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "eval", "test")


def default_num_workers() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(16, max(1, available))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tokenized AudioCaps caption-only datasets for ELF."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("outputs/manifests/audiocaps_v1"),
        help="Directory containing train.jsonl, eval.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/processed/audiocaps_caption_only"),
        help="Output root for audit JSONL and tokenized Hugging Face datasets.",
    )
    parser.add_argument(
        "--tokenizer",
        default="t5-small",
        help="Tokenizer name; must match ELF encoder_model_name (default: t5-small).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=48,
        help="Maximum tokens including EOS (default: 48).",
    )
    parser.add_argument(
        "--num-workers",
        "--n-workers",
        type=int,
        default=0,
        help=(
            "CPU processes for tokenization/save; 0 selects min(16, CPU count) "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=1000,
        help="Examples per tokenizer map batch (default: 1000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Only create audit JSONL files; intended for lightweight checks/tests.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum skipped examples retained per reason in summary.json.",
    )
    return parser.parse_args()


def normalize_caption(value: str) -> str:
    return " ".join(value.split())


def add_skip(summary: dict[str, Any], reason: str, example: str, limit: int) -> None:
    summary["skipped"][reason] = summary["skipped"].get(reason, 0) + 1
    examples = summary["skipped_examples"].setdefault(reason, [])
    if len(examples) < limit:
        examples.append(example)


def validate_manifest_record(
    record: Any,
    split: str,
    line_number: int,
    summary: dict[str, Any],
    max_examples: int,
) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        add_skip(summary, "non_object_record", f"line {line_number}", max_examples)
        return []

    youtube_id = record.get("youtube_id")
    if not isinstance(youtube_id, str) or not youtube_id.strip():
        add_skip(summary, "invalid_youtube_id", f"line {line_number}", max_examples)
        return []
    youtube_id = youtube_id.strip()

    captions = record.get("captions")
    if not isinstance(captions, list) or not captions:
        add_skip(
            summary,
            "missing_caption_list",
            f"line {line_number}: {youtube_id}",
            max_examples,
        )
        return []

    audio_path = record.get("audio_path")
    start_time = record.get("start_time")
    source_split = record.get("source_split", split)
    caption_ids = record.get("audiocap_ids")
    if not isinstance(caption_ids, list):
        caption_ids = []

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for caption_index, raw_caption in enumerate(captions):
        if not isinstance(raw_caption, str):
            add_skip(
                summary,
                "non_string_caption",
                f"line {line_number}, caption {caption_index}: {youtube_id}",
                max_examples,
            )
            continue
        caption = normalize_caption(raw_caption)
        if not caption:
            add_skip(
                summary,
                "empty_caption",
                f"line {line_number}, caption {caption_index}: {youtube_id}",
                max_examples,
            )
            continue
        if caption in seen:
            add_skip(
                summary,
                "duplicate_caption_within_audio",
                f"line {line_number}: {youtube_id}: {caption}",
                max_examples,
            )
            continue
        seen.add(caption)

        caption_id = caption_ids[caption_index] if caption_index < len(caption_ids) else None
        examples.append(
            {
                "index": -1,
                "caption": caption,
                "youtube_id": youtube_id,
                "audio_id": record.get("audio_id", f"Y{youtube_id}"),
                "audio_path": audio_path,
                "start_time": start_time,
                "split": split,
                "source_split": source_split,
                "caption_index": caption_index,
                "audiocap_id": caption_id,
            }
        )
    if not examples:
        add_skip(
            summary,
            "audio_without_valid_caption",
            f"line {line_number}: {youtube_id}",
            max_examples,
        )
    return examples


def load_and_expand_manifest(
    path: Path,
    split: str,
    max_examples: int,
    tqdm_class,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary: dict[str, Any] = {
        "split": split,
        "manifest": str(path),
        "manifest_records": 0,
        "caption_examples": 0,
        "unique_audio": 0,
        "skipped": {},
        "skipped_examples": {},
    }
    examples: list[dict[str, Any]] = []
    audio_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        iterator = tqdm_class(stream, desc=f"Read/expand {split}", unit="records")
        for line_number, line in enumerate(iterator, start=1):
            if not line.strip():
                add_skip(summary, "blank_lines", f"line {line_number}", max_examples)
                continue
            summary["manifest_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                add_skip(
                    summary,
                    "invalid_json",
                    f"line {line_number}: {exc}",
                    max_examples,
                )
                continue
            expanded = validate_manifest_record(
                record, split, line_number, summary, max_examples
            )
            for example in expanded:
                example["index"] = len(examples)
                examples.append(example)
                audio_ids.add(example["youtube_id"])

    summary["caption_examples"] = len(examples)
    summary["unique_audio"] = len(audio_ids)
    return examples, summary


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], tqdm_class) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    records_list = records if isinstance(records, list) else list(records)
    with temporary.open("w", encoding="utf-8") as stream:
        for record in tqdm_class(
            records_list, desc=f"Write {path.name}", unit="captions"
        ):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def check_safe_output_path(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    forbidden = {Path("/"), Path.home(), Path.cwd().absolute()}
    if absolute in forbidden:
        raise ValueError(f"Refusing to use broad output directory: {absolute}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    check_safe_output_path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def tokenize_and_save(
    split: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    tokenizer,
    max_length: int,
    num_workers: int,
    batch_size: int,
) -> dict[str, Any]:
    from datasets import Dataset

    dataset = Dataset.from_list(examples)
    effective_workers = min(num_workers, len(dataset))
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The selected tokenizer does not define eos_token_id")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        encoded = tokenizer(
            batch["caption"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = encoded["input_ids"]
        return {
            "input_ids": input_ids,
            "sequence_length": [len(ids) for ids in input_ids],
        }

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=effective_workers,
        desc=f"Tokenize {split} with {effective_workers} CPU workers",
    )
    lengths = tokenized["sequence_length"]
    if any(length <= 0 or length > max_length for length in lengths):
        raise ValueError(f"{split} contains an invalid token sequence length")
    missing_eos = sum(ids[-1] != eos_token_id for ids in tokenized["input_ids"])
    if missing_eos:
        raise ValueError(
            f"{split} has {missing_eos} token sequences that do not end with EOS"
        )

    split_dir = output_dir / "hf_dataset" / split
    split_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(split_dir), num_proc=effective_workers)
    return {
        "hf_dataset": str(split_dir),
        "tokenized_examples": len(tokenized),
        "tokenization_workers": effective_workers,
        "min_tokens": min(lengths) if lengths else None,
        "mean_tokens": (sum(lengths) / len(lengths)) if lengths else None,
        "max_tokens": max(lengths) if lengths else None,
        "examples_at_max_length": sum(length == max_length for length in lengths),
        "missing_eos": missing_eos,
    }


def load_runtime_dependencies():
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Missing tqdm. Install data dependencies with "
            "`pip install -r requirements-data.txt`."
        ) from exc
    return tqdm


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing transformers. Install data dependencies with "
            "`pip install -r requirements-data.txt`."
        ) from exc
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing datasets. Install data dependencies with "
            "`pip install -r requirements-data.txt`."
        ) from exc
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def main() -> int:
    args = parse_args()
    if args.max_length < 2:
        print("ERROR: --max-length must be >= 2", file=sys.stderr)
        return 2
    if args.num_workers < 0:
        print("ERROR: --num-workers must be >= 0", file=sys.stderr)
        return 2
    if args.tokenize_batch_size <= 0 or args.max_examples < 0:
        print(
            "ERROR: --tokenize-batch-size must be > 0 and --max-examples >= 0",
            file=sys.stderr,
        )
        return 2

    workers = args.num_workers or default_num_workers()
    if workers > 1:
        # Avoid one Rust-tokenizer thread pool per process and CPU oversubscription.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        tqdm_class = load_runtime_dependencies()
        tokenizer = None if args.skip_tokenization else load_tokenizer(args.tokenizer)
        prepare_output_dir(args.output_dir, args.overwrite)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Manifest directory: {args.manifest_dir}")
    print(f"Output directory:   {args.output_dir}")
    print(f"CPU workers:        {workers}")
    print(f"Tokenizer:          {args.tokenizer}")
    print(f"Maximum length:     {args.max_length} tokens (including EOS)")

    all_examples: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    try:
        for split in SPLITS:
            manifest_path = args.manifest_dir / f"{split}.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            examples, split_summary = load_and_expand_manifest(
                manifest_path, split, args.max_examples, tqdm_class
            )
            if not examples:
                raise ValueError(f"No valid captions found in {manifest_path}")
            write_jsonl(args.output_dir / f"{split}.jsonl", examples, tqdm_class)
            all_examples[split] = examples
            summaries.append(split_summary)

        if tokenizer is not None:
            for split, examples in all_examples.items():
                token_summary = tokenize_and_save(
                    split,
                    examples,
                    args.output_dir,
                    tokenizer,
                    args.max_length,
                    workers,
                    args.tokenize_batch_size,
                )
                next(item for item in summaries if item["split"] == split).update(
                    token_summary
                )
            tokenizer.save_pretrained(args.output_dir / "tokenizer")
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = {
        "manifest_dir": str(args.manifest_dir),
        "output_dir": str(args.output_dir),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "num_workers": workers,
        "tokenized": not args.skip_tokenization,
        "splits": summaries,
        "passed": True,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Preparation complete. Summary: {report_path}")
    if not args.skip_tokenization:
        print(f"ELF train data_path: {args.output_dir / 'hf_dataset' / 'train'}")
        print(f"ELF eval_data_path: {args.output_dir / 'hf_dataset' / 'eval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
