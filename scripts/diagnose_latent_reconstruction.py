#!/usr/bin/env python3
"""Experiment 1: decode clean T5 latents with raw and EMA ELF parameters."""

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
            "outputs/diagnostics/elf_caption_overfit_100_ema099/01_latent_reconstruction"
        ),
    )
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    runtime = common.prepare_runtime(args, [args.checkpoint_step])
    config, tokenizer, dataset, references, encoder, model, tqdm, device = runtime
    checkpoint_path = args.checkpoint_root / f"checkpoint_{args.checkpoint_step}"
    checkpoint = common.load_checkpoint(checkpoint_path)
    report = {"checkpoint": str(checkpoint_path), "variants": {}}
    rows = []

    for variant in ("raw", "ema"):
        common.load_weight_variant(model, checkpoint, variant)
        metrics = common.run_reconstruction_variant(
            model=model,
            encoder=encoder,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            references=references,
            output_path=args.output_dir / variant / "reconstructions.jsonl",
            tqdm_class=tqdm,
        )
        report["variants"][variant] = metrics
        rows.append(
            common.flat_metric_row(
                {"checkpoint_step": args.checkpoint_step, "variant": variant},
                metrics,
            )
        )

    common.write_json(args.output_dir / "results.json", report)
    common.write_csv(args.output_dir / "metrics.csv", rows)
    common.write_json(
        args.output_dir / "run_manifest.json",
        common.runtime_manifest(args, device, [args.checkpoint_step]),
    )


def main() -> int:
    try:
        args = parse_args()
        run(args)
        print(f"Latent reconstruction complete: {args.output_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
