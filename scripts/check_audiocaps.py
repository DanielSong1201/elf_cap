#!/usr/bin/env python3
"""Validate an AudioCaps-style WAV/CSV dataset before preprocessing.

Expected default layout::

    data/AudioCaps_CVSSP/
      train/<one CSV file and Y<youtube_id>.wav files>
      val/<one CSV file and Y<youtube_id>.wav files>
      test/<one CSV file and Y<youtube_id>.wav files>

Multiple CSV rows may share a youtube_id (for example, multiple reference
captions in validation/test). The script never changes the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check AudioCaps split CSVs and Y<youtube_id>.wav files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/AudioCaps_CVSSP"),
        help="Dataset root (default: data/AudioCaps_CVSSP).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split directory names to check (default: train val test).",
    )
    parser.add_argument(
        "--id-column",
        default="youtube_id",
        help="CSV column containing the YouTube ID (default: youtube_id).",
    )
    parser.add_argument(
        "--caption-column",
        default="caption",
        help="CSV caption column (default: caption).",
    )
    parser.add_argument(
        "--wav-prefix",
        default="Y",
        help="Prefix before youtube_id in WAV filenames (default: Y).",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Optional CSV filename used in every split; otherwise require exactly one *.csv.",
    )
    parser.add_argument(
        "--skip-wav-header-check",
        action="store_true",
        help="Only check names/mappings; do not open WAV headers.",
    )
    parser.add_argument(
        "--wav-check-limit",
        type=int,
        default=0,
        help="Check at most N matched WAV headers per split; 0 checks all (default: 0).",
    )
    parser.add_argument(
        "--allow-extra-wav",
        action="store_true",
        help="Report WAV files without CSV rows as warnings instead of errors.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum example paths/rows retained per issue type (default: 10).",
    )
    return parser.parse_args()


def add_issue(result: dict[str, Any], level: str, category: str, message: str, limit: int) -> None:
    result[f"{level}_count"] += 1
    examples = result[f"{level}_examples"].setdefault(category, [])
    if len(examples) < limit:
        examples.append(message)


def locate_csv(split_dir: Path, csv_name: str | None) -> tuple[Path | None, str | None]:
    if csv_name:
        candidate = split_dir / csv_name
        if candidate.is_file():
            return candidate, None
        return None, f"CSV does not exist: {candidate}"

    candidates = sorted(split_dir.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, f"No *.csv file found directly under {split_dir}"
    names = ", ".join(path.name for path in candidates)
    return None, f"Expected one CSV under {split_dir}, found {len(candidates)}: {names}"


def open_csv_dict_reader(csv_path: Path):
    stream = csv_path.open("r", encoding="utf-8-sig", newline="")
    sample = stream.read(8192)
    stream.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    return stream, csv.DictReader(stream, dialect=dialect)


def check_wav_header(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()

    if channels <= 0 or sample_rate <= 0 or frames <= 0 or sample_width <= 0:
        raise ValueError(
            f"invalid header values: channels={channels}, rate={sample_rate}, "
            f"frames={frames}, sample_width={sample_width}"
        )
    if compression != "NONE":
        raise ValueError(f"compressed WAV is not PCM: comptype={compression}")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "frames": frames,
        "sample_width_bytes": sample_width,
        "duration_seconds": frames / sample_rate,
    }


def summarize_numbers(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def check_split(split_dir: Path, split: str, args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": split,
        "directory": str(split_dir),
        "csv": None,
        "csv_rows": 0,
        "valid_caption_rows": 0,
        "unique_youtube_ids": 0,
        "unique_captions": 0,
        "wav_files": 0,
        "matched_wav_files": 0,
        "checked_wav_headers": 0,
        "missing_wav_files": 0,
        "extra_wav_files": 0,
        "captions_per_audio": None,
        "wav_duration_seconds": None,
        "sample_rates": {},
        "channel_counts": {},
        "sample_width_bytes": {},
        "error_count": 0,
        "warning_count": 0,
        "error_examples": {},
        "warning_examples": {},
    }

    if not split_dir.is_dir():
        add_issue(result, "error", "missing_split", f"Split directory does not exist: {split_dir}", args.max_examples)
        return result

    csv_path, csv_error = locate_csv(split_dir, args.csv_name)
    if csv_error:
        add_issue(result, "error", "csv", csv_error, args.max_examples)
        return result
    assert csv_path is not None
    result["csv"] = str(csv_path)

    captions_by_id: dict[str, list[str]] = defaultdict(list)
    seen_pairs: Counter[tuple[str, str]] = Counter()
    try:
        stream, reader = open_csv_dict_reader(csv_path)
        with stream:
            fieldnames = reader.fieldnames or []
            if args.id_column not in fieldnames or args.caption_column not in fieldnames:
                add_issue(
                    result,
                    "error",
                    "csv_columns",
                    f"Required columns {args.id_column!r}, {args.caption_column!r}; found {fieldnames}",
                    args.max_examples,
                )
                return result

            for row_number, row in enumerate(reader, start=2):
                result["csv_rows"] += 1
                youtube_id = (row.get(args.id_column) or "").strip()
                caption = (row.get(args.caption_column) or "").strip()
                if not youtube_id:
                    add_issue(result, "error", "empty_youtube_id", f"row {row_number}", args.max_examples)
                    continue
                if any(char in youtube_id for char in ("/", "\\", "\x00")):
                    add_issue(
                        result,
                        "error",
                        "unsafe_youtube_id",
                        f"row {row_number}: {youtube_id!r}",
                        args.max_examples,
                    )
                    continue
                if not caption:
                    add_issue(
                        result,
                        "error",
                        "empty_caption",
                        f"row {row_number}, youtube_id={youtube_id!r}",
                        args.max_examples,
                    )
                    continue

                result["valid_caption_rows"] += 1
                captions_by_id[youtube_id].append(caption)
                seen_pairs[(youtube_id, caption)] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        add_issue(result, "error", "csv_read", f"{csv_path}: {exc}", args.max_examples)
        return result

    duplicate_pairs = [(pair, count) for pair, count in seen_pairs.items() if count > 1]
    for (youtube_id, caption), count in duplicate_pairs:
        add_issue(
            result,
            "warning",
            "duplicate_caption_row",
            f"youtube_id={youtube_id!r}, repeated {count} times: {caption!r}",
            args.max_examples,
        )

    result["unique_youtube_ids"] = len(captions_by_id)
    result["unique_captions"] = len(seen_pairs)
    captions_per_audio = [len(captions) for captions in captions_by_id.values()]
    result["captions_per_audio"] = summarize_numbers([float(value) for value in captions_per_audio])
    if not captions_by_id:
        add_issue(
            result,
            "error",
            "empty_split",
            f"No valid ({args.id_column}, {args.caption_column}) rows found in {csv_path}",
            args.max_examples,
        )

    wav_paths = sorted(
        path for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    result["wav_files"] = len(wav_paths)
    wav_by_name: dict[str, Path] = {}
    for wav_path in wav_paths:
        if wav_path.name in wav_by_name:
            add_issue(result, "error", "duplicate_wav_name", wav_path.name, args.max_examples)
        wav_by_name[wav_path.name] = wav_path

    expected_names = {
        f"{args.wav_prefix}{youtube_id}.wav": youtube_id for youtube_id in captions_by_id
    }
    missing_names = sorted(set(expected_names) - set(wav_by_name))
    extra_names = sorted(set(wav_by_name) - set(expected_names))
    matched_names = sorted(set(expected_names) & set(wav_by_name))
    result["missing_wav_files"] = len(missing_names)
    result["extra_wav_files"] = len(extra_names)
    result["matched_wav_files"] = len(matched_names)

    for name in missing_names:
        add_issue(
            result,
            "error",
            "missing_wav",
            f"youtube_id={expected_names[name]!r} expects {split_dir / name}",
            args.max_examples,
        )
    for name in extra_names:
        level = "warning" if args.allow_extra_wav else "error"
        add_issue(result, level, "extra_wav", str(wav_by_name[name]), args.max_examples)

    if not args.skip_wav_header_check:
        names_to_check = matched_names
        if args.wav_check_limit > 0:
            names_to_check = names_to_check[: args.wav_check_limit]

        durations: list[float] = []
        sample_rates: Counter[int] = Counter()
        channels: Counter[int] = Counter()
        sample_widths: Counter[int] = Counter()
        for name in names_to_check:
            wav_path = wav_by_name[name]
            try:
                metadata = check_wav_header(wav_path)
            except (OSError, EOFError, wave.Error, ValueError) as exc:
                add_issue(result, "error", "invalid_wav", f"{wav_path}: {exc}", args.max_examples)
                continue
            result["checked_wav_headers"] += 1
            durations.append(float(metadata["duration_seconds"]))
            sample_rates[int(metadata["sample_rate"])] += 1
            channels[int(metadata["channels"])] += 1
            sample_widths[int(metadata["sample_width_bytes"])] += 1

        result["wav_duration_seconds"] = summarize_numbers(durations)
        result["sample_rates"] = {str(key): value for key, value in sorted(sample_rates.items())}
        result["channel_counts"] = {str(key): value for key, value in sorted(channels.items())}
        result["sample_width_bytes"] = {str(key): value for key, value in sorted(sample_widths.items())}

    return result


def print_split_summary(result: dict[str, Any]) -> None:
    print(f"\n[{result['split']}]")
    print(f"  directory:          {result['directory']}")
    print(f"  csv:                {result['csv'] or '-'}")
    print(f"  csv rows:           {result['csv_rows']}")
    print(f"  valid caption rows: {result['valid_caption_rows']}")
    print(f"  unique youtube IDs: {result['unique_youtube_ids']}")
    print(f"  WAV files:          {result['wav_files']}")
    print(f"  matched WAV files:  {result['matched_wav_files']}")
    print(f"  missing WAV files:  {result['missing_wav_files']}")
    print(f"  extra WAV files:    {result['extra_wav_files']}")
    print(f"  WAV headers checked:{result['checked_wav_headers']:>5}")
    if result["wav_duration_seconds"]:
        duration = result["wav_duration_seconds"]
        print(
            "  duration seconds:   "
            f"min={duration['min']:.3f}, mean={duration['mean']:.3f}, max={duration['max']:.3f}"
        )
    if result["sample_rates"]:
        print(f"  sample rates:       {result['sample_rates']}")
        print(f"  channel counts:     {result['channel_counts']}")
    print(f"  errors/warnings:    {result['error_count']}/{result['warning_count']}")

    for level in ("error", "warning"):
        examples = result[f"{level}_examples"]
        for category, messages in examples.items():
            print(f"  {level.upper()} {category}:")
            for message in messages:
                print(f"    - {message}")


def main() -> int:
    args = parse_args()
    if args.wav_check_limit < 0:
        print("ERROR: --wav-check-limit must be >= 0", file=sys.stderr)
        return 2
    if args.max_examples < 0:
        print("ERROR: --max-examples must be >= 0", file=sys.stderr)
        return 2

    data_root = args.data_root.expanduser()
    print(f"AudioCaps dataset check: {data_root}")
    print("This script is read-only; it will not modify audio or CSV files.")

    results = [check_split(data_root / split, split, args) for split in args.splits]
    for result in results:
        print_split_summary(result)

    total_errors = sum(result["error_count"] for result in results)
    total_warnings = sum(result["warning_count"] for result in results)
    report = {
        "data_root": str(data_root),
        "splits": results,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "passed": total_errors == 0,
    }

    if args.report:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {report_path}")

    status = "PASSED" if total_errors == 0 else "FAILED"
    print(f"\nResult: {status} ({total_errors} errors, {total_warnings} warnings)")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
