#!/usr/bin/env python3
"""Experiment 3: compare ODE/SDE and SC-CFG 1/3 using EMA parameters."""

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
            "outputs/diagnostics/elf_caption_overfit_100_ema099/03_sampler_sweep"
        ),
    )
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[400, 1000])
    parser.add_argument("--self-cond-cfg-scales", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument(
        "--sampling-methods", nargs="+", choices=("ode", "sde"), default=["ode", "sde"]
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    runtime = common.prepare_runtime(args, args.checkpoint_steps)
    config, tokenizer, _dataset, references, _encoder, model, tqdm, device = runtime
    combinations = [
        (method, scale)
        for method in args.sampling_methods
        for scale in args.self_cond_cfg_scales
    ]
    report = {"checkpoints": {}}
    rows = []

    for step in tqdm(args.checkpoint_steps, desc="Sampler checkpoints", unit="checkpoints"):
        checkpoint_path = args.checkpoint_root / f"checkpoint_{step}"
        checkpoint = common.load_checkpoint(checkpoint_path)
        common.load_weight_variant(model, checkpoint, "ema")
        step_report = {}
        for method, scale in tqdm(combinations, desc=f"Checkpoint {step} samplers", unit="configs"):
            run_name = f"{method}_steps{args.sampling_steps}_sccfg{scale:g}"
            sampler = common.sampling_config(method, scale, args)
            metrics = common.generation_run(
                model=model,
                tokenizer=tokenizer,
                config=config,
                sc=sampler,
                args=args,
                references=references,
                output_path=(
                    args.output_dir / f"checkpoint_{step}" / run_name / "generations.jsonl"
                ),
                tqdm_class=tqdm,
            )
            step_report[run_name] = metrics
            rows.append(
                common.flat_metric_row(
                    {
                        "checkpoint_step": step,
                        "parameter_variant": "ema",
                        "sampling_method": method,
                        "sampling_steps": args.sampling_steps,
                        "self_cond_cfg_scale": scale,
                    },
                    metrics,
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
        print(f"Sampler sweep complete: {args.output_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
