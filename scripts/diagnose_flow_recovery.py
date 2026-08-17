#!/usr/bin/env python3
"""Recover clean caption latents from controlled noise with the ELF flow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import caption_diagnostics_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_runtime_arguments(
        parser,
        default_output_dir=Path(
            "outputs/diagnostics/elf_caption_overfit_100_ema099/04_flow_recovery"
        ),
    )
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[1000])
    parser.add_argument("--variants", nargs="+", choices=("raw", "ema"), default=["raw"])
    parser.add_argument(
        "--start-times",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
        help="Known-signal fractions t in z_t=t*x0+(1-t)*noise.",
    )
    parser.add_argument(
        "--sampling-method", choices=("ode", "sde"), default="ode"
    )
    parser.add_argument(
        "--time-schedule", choices=("uniform", "logit_normal"), default="uniform"
    )
    parser.add_argument("--self-cond-cfg-scale", type=float, default=1.0)
    return parser.parse_args()


def validate_start_times(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("--start-times must contain at least one value")
    invalid = [value for value in values if not 0.0 <= value <= 1.0]
    if invalid:
        raise ValueError(f"start times must be in [0, 1], got {invalid}")
    return sorted(set(float(value) for value in values))


def build_recovery_steps(
    start_time: float,
    num_steps: int,
    schedule: str,
    config,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build exactly num_steps integration intervals from start_time to 1."""
    if start_time == 1.0:
        return torch.ones((1,), device=device, dtype=dtype)
    if schedule == "uniform":
        return torch.linspace(start_time, 1.0, num_steps + 1, device=device, dtype=dtype)
    base = common.get_sampling_steps(
        n_steps=num_steps,
        time_schedule="logit_normal",
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        device=device,
        dtype=dtype,
    )
    return start_time + (1.0 - start_time) * base


def masked_latent_metrics(
    predicted: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, list[float]]:
    """Return per-example latent errors over valid caption positions only."""
    mask = attention_mask.to(device=target.device, dtype=target.dtype).unsqueeze(-1)
    difference = (predicted - target) * mask
    valid_dimensions = attention_mask.sum(dim=1).clamp(min=1).to(target.dtype)
    valid_dimensions = valid_dimensions * target.shape[-1]
    mse = difference.square().sum(dim=(1, 2)) / valid_dimensions
    target_masked = (target * mask).flatten(1)
    predicted_masked = (predicted * mask).flatten(1)
    cosine = F.cosine_similarity(predicted_masked, target_masked, dim=1, eps=1e-8)
    relative_l2 = difference.flatten(1).norm(dim=1) / target_masked.norm(dim=1).clamp(min=1e-8)
    return {
        "mse": mse.detach().float().cpu().tolist(),
        "cosine_similarity": cosine.detach().float().cpu().tolist(),
        "relative_l2_error": relative_l2.detach().float().cpu().tolist(),
        "target_l2_norm": target_masked.norm(dim=1).detach().float().cpu().tolist(),
        "predicted_l2_norm": predicted_masked.norm(dim=1).detach().float().cpu().tolist(),
    }


def update_distribution_sums(
    totals: dict[str, float], prefix: str, value: torch.Tensor, attention_mask: torch.Tensor
) -> None:
    mask = attention_mask.bool().unsqueeze(-1).expand_as(value)
    selected = value.detach().float()[mask]
    totals[f"{prefix}_sum"] += float(selected.sum().item())
    totals[f"{prefix}_squared_sum"] += float(selected.square().sum().item())
    totals[f"{prefix}_count"] += int(selected.numel())


def finish_distribution(totals: dict[str, float], prefix: str) -> dict[str, float]:
    count = max(1, int(totals[f"{prefix}_count"]))
    value_mean = totals[f"{prefix}_sum"] / count
    second_moment = totals[f"{prefix}_squared_sum"] / count
    variance = max(0.0, second_moment - value_mean * value_mean)
    return {
        "mean": value_mean,
        "std": math.sqrt(variance),
        "rms": math.sqrt(max(0.0, second_moment)),
    }


def decode_batch(latent, model, tokenizer, config, scale: float):
    pad_id = common.get_pad_token_id(tokenizer, config.pad_token)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token")
    predicted_ids = common._dlm_decode_batch(latent, model, 1.0, config, scale)
    display_ids = common.mask_after_eos(predicted_ids, eos_id, pad_id)
    texts = [
        tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True)
        for row in display_ids
    ]
    return predicted_ids.detach().cpu(), texts


def token_metrics(
    predicted_ids: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_id: int,
    predicted_texts: list[str],
    target_texts: list[str],
) -> dict[str, float]:
    valid = attention_mask.bool()
    correct = predicted_ids.eq(target_ids)
    exact_tokens = (correct | ~valid).all(dim=1)
    predicted_eos = common.first_eos_positions(predicted_ids, eos_id)
    target_eos = common.first_eos_positions(target_ids, eos_id)
    count = max(1, target_ids.shape[0])
    return {
        "token_accuracy": float((correct & valid).sum().item() / max(1, valid.sum().item())),
        "exact_token_reconstruction_rate": float(exact_tokens.sum().item() / count),
        "exact_text_reconstruction_rate": sum(
            common.normalize_text(predicted) == common.normalize_text(target)
            for predicted, target in zip(predicted_texts, target_texts)
        ) / count,
        "eos_emission_rate": float(predicted_eos.ge(0).sum().item() / count),
        "eos_position_accuracy": float(predicted_eos.eq(target_eos).sum().item() / count),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def time_name(start_time: float) -> str:
    return f"t_{start_time:.2f}".replace(".", "p")


@torch.inference_mode()
def run_one_start_time(
    *, args, start_time, checkpoint_step, variant, model, encoder, tokenizer,
    dataset, references, config, device, tqdm,
) -> dict[str, Any]:
    pad_id = common.get_pad_token_id(tokenizer, config.pad_token)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token")
    loader = common.get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )
    sampler = common.sampling_config(args.sampling_method, args.self_cond_cfg_scale, args)
    sampler.time_schedule = args.time_schedule
    model_dtype = next(model.parameters()).dtype

    target_ids_all = []
    attention_all = []
    direct_ids_all = []
    recovered_ids_all = []
    target_texts: list[str] = []
    direct_texts: list[str] = []
    recovered_texts: list[str] = []
    latent_before_values = {
        "mse": [], "cosine_similarity": [], "relative_l2_error": [],
        "target_l2_norm": [], "predicted_l2_norm": [],
    }
    latent_after_values = {key: [] for key in latent_before_values}
    distribution_totals = {
        "clean_sum": 0.0,
        "clean_squared_sum": 0.0,
        "clean_count": 0,
        "recovered_sum": 0.0,
        "recovered_squared_sum": 0.0,
        "recovered_count": 0,
    }
    records: list[dict[str, Any]] = []
    offset = 0

    description = f"Recover ckpt{checkpoint_step} {variant} t={start_time:g}"
    for batch_index, batch in enumerate(tqdm(loader, desc=description, unit="batches")):
        target_ids = common.batch_value_to_tensor(
            batch["input_ids"], device, dtype=torch.long
        )
        attention = common.batch_value_to_tensor(
            batch["attention_mask"], device, dtype=torch.float32
        )
        encoder_attention = common.batch_value_to_tensor(
            batch["encoder_attention_mask"], device, dtype=torch.float32
        )
        clean_latent = common.encode_text(
            target_ids,
            encoder_attention,
            encoder,
            config.latent_mean,
            config.latent_std,
            use_bf16=config.use_bf16,
        ).to(model_dtype)
        generator = common.reset_random_seed(args.seed + batch_index, device)
        noise = torch.randn(
            clean_latent.shape, device=device, dtype=model_dtype
        ) * config.denoiser_noise_scale
        start_latent = start_time * clean_latent + (1.0 - start_time) * noise

        if math.isclose(start_time, 1.0):
            recovered_latent = clean_latent
        else:
            steps = build_recovery_steps(
                start_time,
                args.sampling_steps,
                args.time_schedule,
                config,
                device,
                model_dtype,
            )
            recovered_latent = common._generate_samples_single_batch(
                model=model,
                generator=generator,
                z=start_latent,
                t_steps=steps,
                cond_seq=None,
                cond_seq_mask=None,
                config=config,
                sampling_config=sampler,
                cfg_scale=1.0,
                self_cond_cfg_scale=args.self_cond_cfg_scale,
            )

        direct_ids, direct_batch_texts = decode_batch(
            start_latent, model, tokenizer, config, args.self_cond_cfg_scale
        )
        recovered_ids, recovered_batch_texts = decode_batch(
            recovered_latent, model, tokenizer, config, args.self_cond_cfg_scale
        )
        target_batch_texts = [
            tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=True)
            for row in target_ids
        ]
        before_latent_metrics = masked_latent_metrics(
            start_latent, clean_latent, attention
        )
        after_latent_metrics = masked_latent_metrics(
            recovered_latent, clean_latent, attention
        )
        for key in latent_before_values:
            latent_before_values[key].extend(before_latent_metrics[key])
            latent_after_values[key].extend(after_latent_metrics[key])
        update_distribution_sums(
            distribution_totals, "clean", clean_latent, attention
        )
        update_distribution_sums(
            distribution_totals, "recovered", recovered_latent, attention
        )

        target_cpu = target_ids.detach().cpu()
        attention_cpu = attention.detach().cpu()
        target_ids_all.append(target_cpu)
        attention_all.append(attention_cpu)
        direct_ids_all.append(direct_ids)
        recovered_ids_all.append(recovered_ids)
        target_texts.extend(target_batch_texts)
        direct_texts.extend(direct_batch_texts)
        recovered_texts.extend(recovered_batch_texts)
        for row in range(target_ids.shape[0]):
            records.append(
                {
                    "id": offset + row,
                    "target": target_batch_texts[row],
                    "direct_decode": direct_batch_texts[row],
                    "recovered_decode": recovered_batch_texts[row],
                    "latent_mse_before_flow": before_latent_metrics["mse"][row],
                    "latent_mse_after_flow": after_latent_metrics["mse"][row],
                    "latent_cosine_before_flow": before_latent_metrics["cosine_similarity"][row],
                    "latent_cosine_after_flow": after_latent_metrics["cosine_similarity"][row],
                    "latent_relative_l2_before_flow": before_latent_metrics["relative_l2_error"][row],
                    "latent_relative_l2_after_flow": after_latent_metrics["relative_l2_error"][row],
                }
            )
        offset += target_ids.shape[0]

    target_ids_cat = torch.cat(target_ids_all)
    attention_cat = torch.cat(attention_all)
    direct_ids_cat = torch.cat(direct_ids_all)
    recovered_ids_cat = torch.cat(recovered_ids_all)
    direct_metrics = common.generation_metrics(direct_texts, references)
    direct_metrics.update(
        token_metrics(
            direct_ids_cat, target_ids_cat, attention_cat, eos_id,
            direct_texts, target_texts,
        )
    )
    recovered_metrics = common.generation_metrics(recovered_texts, references)
    recovered_metrics.update(
        token_metrics(
            recovered_ids_cat, target_ids_cat, attention_cat, eos_id,
            recovered_texts, target_texts,
        )
    )
    output_dir = args.output_dir / f"checkpoint_{checkpoint_step}" / variant / time_name(start_time)
    common.write_jsonl(output_dir / "recoveries.jsonl", records)
    return {
        "checkpoint_step": checkpoint_step,
        "variant": variant,
        "start_time": start_time,
        "known_signal_fraction": start_time,
        "noise_fraction": 1.0 - start_time,
        "sampling_method": args.sampling_method,
        "time_schedule": args.time_schedule,
        "sampling_steps": args.sampling_steps,
        "self_cond_cfg_scale": args.self_cond_cfg_scale,
        "latent_before_flow": {
            key: mean(values) for key, values in latent_before_values.items()
        },
        "latent_after_flow": {
            key: mean(values) for key, values in latent_after_values.items()
        },
        "latent_distribution": {
            "clean": finish_distribution(distribution_totals, "clean"),
            "recovered": finish_distribution(distribution_totals, "recovered"),
        },
        "direct_decode": direct_metrics,
        "recovered_decode": recovered_metrics,
        "recoveries_file": str(output_dir / "recoveries.jsonl"),
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in result.items()
        if not isinstance(value, dict)
    }
    for section in (
        "latent_before_flow", "latent_after_flow", "direct_decode", "recovered_decode"
    ):
        for key, value in result[section].items():
            if isinstance(value, (str, int, float, bool)):
                row[f"{section}_{key}"] = value
    for latent_name, metrics in result["latent_distribution"].items():
        for key, value in metrics.items():
            row[f"latent_distribution_{latent_name}_{key}"] = value
    return row


def run(args: argparse.Namespace) -> None:
    args.start_times = validate_start_times(args.start_times)
    runtime = common.prepare_runtime(args, args.checkpoint_steps)
    config, tokenizer, dataset, references, encoder, model, tqdm, device = runtime
    report: dict[str, Any] = {"runs": []}
    rows = []
    for step in tqdm(args.checkpoint_steps, desc="Recovery checkpoints", unit="checkpoints"):
        checkpoint = common.load_checkpoint(args.checkpoint_root / f"checkpoint_{step}")
        for variant in args.variants:
            common.load_weight_variant(model, checkpoint, variant)
            for start_time in tqdm(args.start_times, desc=f"Checkpoint {step} {variant}", unit="times"):
                result = run_one_start_time(
                    args=args,
                    start_time=start_time,
                    checkpoint_step=step,
                    variant=variant,
                    model=model,
                    encoder=encoder,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    references=references,
                    config=config,
                    device=device,
                    tqdm=tqdm,
                )
                report["runs"].append(result)
                rows.append(flatten_result(result))
    common.write_json(args.output_dir / "results.json", report)
    common.write_csv(args.output_dir / "metrics.csv", rows)
    manifest = common.runtime_manifest(args, device, args.checkpoint_steps)
    manifest.update(
        {
            "variants": args.variants,
            "start_times": args.start_times,
            "sampling_method": args.sampling_method,
            "time_schedule": args.time_schedule,
            "self_cond_cfg_scale": args.self_cond_cfg_scale,
        }
    )
    common.write_json(args.output_dir / "run_manifest.json", manifest)


def main() -> int:
    try:
        args = parse_args()
        run(args)
        print(f"Flow recovery experiment complete: {args.output_dir}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
