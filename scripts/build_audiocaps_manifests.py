#!/usr/bin/env python3
"""Build JSONL manifests from AudioCaps metadata and CVSSP WAV files.

The audio and metadata roots are deliberately separate: this allows the
CVSSP bundle to be used only as an audio cache while captions come from the
official AudioCaps v1 CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = (("train", "train"), ("val", "eval"), ("test", "test"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/eval/test JSONL manifests for AudioCaps."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/AudioCaps_CVSSP"),
        help="Root containing train/val/test WAV directories.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help=(
            "Caption metadata root. It may contain <split>.csv (official layout) "
            "or <split>/*.csv (CVSSP layout). Defaults to --audio-root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manifests/audiocaps"),
        help="Directory for train.jsonl, eval.jsonl, test.jsonl and summary.json.",
    )
    parser.add_argument("--id-column", default="youtube_id")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--start-time-column", default="start_time")
    parser.add_argument("--caption-id-column", default="audiocap_id")
    parser.add_argument("--wav-prefix", default="Y")
    parser.add_argument("--wav-extension", default=".wav")
    parser.add_argument(
        "--absolute-audio-paths",
        action="store_true",
        help="Store absolute paths instead of paths relative to the current directory.",
    )
    parser.add_argument(
        "--skip-wav-validation",
        action="store_true",
        help="Check WAV existence only; do not validate its header.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum skipped examples retained per reason in summary.json.",
    )
    return parser.parse_args()


def open_csv_dict_reader(csv_path: Path):
    stream = csv_path.open("r", encoding="utf-8-sig", newline="")
    sample = stream.read(8192)
    stream.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    return stream, csv.DictReader(stream, dialect=dialect)


def locate_csv(metadata_root: Path, split: str) -> Path:
    flat_csv = metadata_root / f"{split}.csv"
    if flat_csv.is_file():
        return flat_csv

    split_dir = metadata_root / split
    candidates = sorted(split_dir.glob("*.csv")) if split_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No metadata found at {flat_csv} or directly under {split_dir}"
        )
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Expected one CSV under {split_dir}, found: {names}")


def normalize_start_time(value: str) -> int | float | str | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def valid_wav(path: Path) -> tuple[bool, str | None]:
    try:
        if path.stat().st_size <= 0:
            return False, "empty_wav"
        with wave.open(str(path), "rb") as wav_file:
            if (
                wav_file.getnchannels() <= 0
                or wav_file.getframerate() <= 0
                or wav_file.getnframes() <= 0
            ):
                return False, "invalid_wav_header"
    except (OSError, EOFError, wave.Error):
        return False, "invalid_wav_header"
    return True, None


def add_skip(
    summary: dict[str, Any], reason: str, example: str, max_examples: int
) -> None:
    summary["skipped"][reason] = summary["skipped"].get(reason, 0) + 1
    examples = summary["skipped_examples"].setdefault(reason, [])
    if len(examples) < max_examples:
        examples.append(example)


def display_audio_path(path: Path, absolute: bool) -> str:
    # abspath keeps the user-facing symlink path, while Path.resolve() would
    # replace data/AudioCaps_CVSSP with its server-specific link target.
    absolute_path = Path(os.path.abspath(path))
    if absolute:
        return str(absolute_path)
    return os.path.relpath(absolute_path, Path.cwd().absolute())


def read_caption_groups(
    csv_path: Path, args: argparse.Namespace, summary: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Read and aggregate valid captions by YouTube ID.

    WAV names do not encode start_time. If one ID refers to multiple start
    times, that ID is marked ambiguous and omitted later instead of silently
    assigning one WAV file to multiple clips.
    """
    groups: dict[str, dict[str, Any]] = {}
    start_values: dict[str, set[str]] = defaultdict(set)

    stream, reader = open_csv_dict_reader(csv_path)
    with stream:
        fields = reader.fieldnames or []
        required = {args.id_column, args.caption_column}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        has_start_time = args.start_time_column in fields
        has_caption_id = args.caption_id_column in fields
        summary["metadata_columns"] = fields

        for row_number, row in enumerate(reader, start=2):
            summary["metadata_rows"] += 1
            youtube_id = (row.get(args.id_column) or "").strip()
            caption = (row.get(args.caption_column) or "").strip()
            if not youtube_id:
                add_skip(
                    summary,
                    "empty_youtube_id_rows",
                    f"row {row_number}",
                    args.max_examples,
                )
                continue
            if any(char in youtube_id for char in ("/", "\\", "\x00")):
                add_skip(
                    summary,
                    "unsafe_youtube_id_rows",
                    f"row {row_number}: {youtube_id!r}",
                    args.max_examples,
                )
                continue
            if not caption:
                add_skip(
                    summary,
                    "empty_caption_rows",
                    f"row {row_number}: {youtube_id}",
                    args.max_examples,
                )
                continue

            raw_start_time = (
                (row.get(args.start_time_column) or "").strip() if has_start_time else ""
            )
            start_values[youtube_id].add(raw_start_time)
            group = groups.setdefault(
                youtube_id,
                {
                    "start_time": normalize_start_time(raw_start_time),
                    "captions": [],
                    "caption_ids": [],
                    "_seen_captions": set(),
                },
            )
            if caption in group["_seen_captions"]:
                add_skip(
                    summary,
                    "duplicate_caption_rows",
                    f"row {row_number}: {youtube_id}: {caption}",
                    args.max_examples,
                )
                continue
            group["_seen_captions"].add(caption)
            group["captions"].append(caption)
            if has_caption_id:
                caption_id = (row.get(args.caption_id_column) or "").strip()
                group["caption_ids"].append(caption_id or None)

    for youtube_id, values in start_values.items():
        if len(values) > 1 and youtube_id in groups:
            groups[youtube_id]["_ambiguous_start_time"] = sorted(values)
    return groups


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_split(
    source_split: str,
    output_split: str,
    audio_root: Path,
    metadata_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    csv_path = locate_csv(metadata_root, source_split)
    audio_dir = audio_root / source_split
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio split directory does not exist: {audio_dir}")

    summary: dict[str, Any] = {
        "source_split": source_split,
        "manifest_split": output_split,
        "metadata_csv": str(csv_path),
        "audio_directory": str(audio_dir),
        "metadata_rows": 0,
        "metadata_columns": [],
        "candidate_audio_ids": 0,
        "written_samples": 0,
        "written_captions": 0,
        "skipped": {},
        "skipped_examples": {},
    }
    groups = read_caption_groups(csv_path, args, summary)
    summary["candidate_audio_ids"] = len(groups)
    records: list[dict[str, Any]] = []

    extension = args.wav_extension
    if not extension.startswith("."):
        extension = f".{extension}"

    # WAV files with no valid metadata group are deliberately excluded too.
    # Reporting them makes a missing-caption condition visible instead of
    # silently ignoring extra audio in the CVSSP cache.
    for wav_path in sorted(audio_dir.iterdir()):
        if not wav_path.is_file() or wav_path.suffix.lower() != extension.lower():
            continue
        if not wav_path.stem.startswith(args.wav_prefix):
            continue
        audio_youtube_id = wav_path.stem[len(args.wav_prefix) :]
        if audio_youtube_id and audio_youtube_id not in groups:
            add_skip(
                summary,
                "audio_without_valid_caption",
                str(wav_path),
                args.max_examples,
            )

    for youtube_id in sorted(groups):
        group = groups[youtube_id]
        if "_ambiguous_start_time" in group:
            add_skip(
                summary,
                "ambiguous_start_time_audio",
                f"{youtube_id}: {group['_ambiguous_start_time']}",
                args.max_examples,
            )
            continue

        wav_path = audio_dir / f"{args.wav_prefix}{youtube_id}{extension}"
        if not wav_path.is_file():
            add_skip(summary, "missing_audio", str(wav_path), args.max_examples)
            continue
        if not args.skip_wav_validation:
            is_valid, reason = valid_wav(wav_path)
            if not is_valid:
                add_skip(
                    summary,
                    reason or "invalid_wav",
                    str(wav_path),
                    args.max_examples,
                )
                continue

        record: dict[str, Any] = {
            "split": output_split,
            "source_split": source_split,
            "audio_id": f"{args.wav_prefix}{youtube_id}",
            "youtube_id": youtube_id,
            "start_time": group["start_time"],
            "audio_path": display_audio_path(wav_path, args.absolute_audio_paths),
            "captions": group["captions"],
            "num_captions": len(group["captions"]),
        }
        if group["caption_ids"]:
            record["audiocap_ids"] = group["caption_ids"]
        records.append(record)
        summary["written_captions"] += len(group["captions"])

    summary["written_samples"] = len(records)
    manifest_path = output_dir / f"{output_split}.jsonl"
    write_jsonl_atomic(manifest_path, records)
    summary["manifest"] = str(manifest_path)
    return summary


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        print("ERROR: --max-examples must be >= 0", file=sys.stderr)
        return 2

    audio_root = args.audio_root.expanduser()
    metadata_root = (args.metadata_root or audio_root).expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio root:    {audio_root}")
    print(f"Metadata root: {metadata_root}")
    print(f"Output dir:    {output_dir}")

    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for source_split, output_split in SPLITS:
        try:
            summary = build_split(
                source_split,
                output_split,
                audio_root,
                metadata_root,
                output_dir,
                args,
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            message = f"{source_split} -> {output_split}: {exc}"
            failures.append(message)
            print(f"ERROR: {message}", file=sys.stderr)
            continue
        summaries.append(summary)
        skipped = sum(summary["skipped"].values())
        print(
            f"{source_split:>5} -> {output_split:<5}: "
            f"{summary['written_samples']} audio, "
            f"{summary['written_captions']} captions, "
            f"{skipped} skipped rows/audio"
        )

    report = {
        "audio_root": str(audio_root),
        "metadata_root": str(metadata_root),
        "output_dir": str(output_dir),
        "splits": summaries,
        "failures": failures,
        "passed": not failures,
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Summary: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
