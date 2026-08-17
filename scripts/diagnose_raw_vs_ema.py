#!/usr/bin/env python3
"""Experiment 2: compare raw and EMA ELF parameters with identical noise."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import caption_diagnostics_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(
        parser,
        default_output_dir=Path(
            "outputs/diagnostics/elf_caption_overfit_100_ema099/02_raw_vs_ema"
        ),
    )
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[400, 1000])
    parser.add_argument("--self-cond-cfg-scale", type=float, default=3.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    runtime = common.prepare_runtime(args, args.checkpoint_steps)
    config, tokenizer, _dataset, references, _encoder, model, tqdm, device = runtime
    sampler = common.sampling_config("sde", args.self_cond_cfg_scale, args)
    report = {"sampling": vars(sampler), "checkpoints": {}}
    rows = []

    for step in tqdm(args.checkpoint_steps, desc="Raw/EMA checkpoints", unit="checkpoints"):
        checkpoint_path = args.checkpoint_root / f"checkpoint_{step}"
        checkpoint = common.load_checkpoint(checkpoint_path)
        step_report = {}
        for variant in ("raw", "ema"):
            common.load_weight_variant(model, checkpoint, variant)
            metrics = common.generation_run(
                model=model,
                tokenizer=tokenizer,
                config=config,
                sc=sampler,
                args=args,
                references=references,
                output_path=(
                    args.output_dir / f"checkpoint_{step}" / variant / "generations.jsonl"
                ),
                tqdm_class=tqdm,
            )
            step_report[variant] = metrics
            rows.append(
                common.flat_metric_row(
                    {"checkpoint_step": step, "variant": variant}, metrics
                )
            )
        report["checkpoints"][str(step)] = step_report

    common.write_json(args.output_dir / "results.json", report)
    common.write_csv(args.output_dir / "metrics.csv", rows)
    common.write_json(
        args.output_dir / "run_manifest.json",
        common.runtime_manifest(args, device, args.checkpoint_steps),
    )


def main() -> int:
    try:
        args = parse_args()
        run(args)
        print(f"Raw/EMA comparison complete: {args.output_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
