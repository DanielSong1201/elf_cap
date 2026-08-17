#!/usr/bin/env python3
"""Shared runtime helpers for ELF caption diagnostic experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ELF_ROOT = REPO_ROOT / "elf"
if str(ELF_ROOT) not in sys.path:
    sys.path.insert(0, str(ELF_ROOT))

from configs.config import SamplingConfig, load_config_from_yaml  # noqa: E402
from modules.model import ELF_models  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset_split  # noqa: E402
from utils.encoder_utils import encode_text  # noqa: E402
from utils.generation_utils import (  # noqa: E402
    _dlm_decode_batch,
    _generate_samples_single_batch,
    mask_after_eos,
)
from utils.sampling_utils import get_sampling_steps  # noqa: E402


WORD_PATTERN = re.compile(r"[\w']+", re.UNICODE)


def add_runtime_arguments(
    parser: argparse.ArgumentParser, *, default_output_dir: Path
) -> None:
    """Add the paths and runtime flags shared by all three entry scripts."""
    default_root = Path("outputs/experiments/elf_caption_overfit_100_ema099")
    parser.add_argument("--experiment-root", type=Path, default=default_root)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--sde-gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cpu", action="store_true")


def normalize_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.lower()))


def has_repeated_bigram(text: str) -> bool:
    tokens = WORD_PATTERN.findall(text.lower())
    bigrams = list(zip(tokens, tokens[1:]))
    return len(tokens) >= 4 and len(bigrams) != len(set(bigrams))


def generation_metrics(generated: list[str], references: list[str]) -> dict[str, Any]:
    normalized_refs = [normalize_text(text) for text in references]
    reference_set = set(normalized_refs)
    normalized = [normalize_text(text) for text in generated]
    nonempty = [(raw, norm) for raw, norm in zip(generated, normalized) if norm]
    similarities: list[float] = []
    closest: list[dict[str, Any]] = []
    for raw, norm in nonempty:
        scores = [
            SequenceMatcher(None, norm, reference, autojunk=False).ratio()
            for reference in normalized_refs
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        similarities.append(scores[best_index])
        closest.append(
            {
                "generated": raw,
                "nearest_reference": references[best_index],
                "similarity": scores[best_index],
            }
        )
    exact_values = {norm for _, norm in nonempty if norm in reference_set}
    word_counts = [len(WORD_PATTERN.findall(raw)) for raw, _ in nonempty]
    denominator = max(1, len(nonempty))
    return {
        "num_generated": len(generated),
        "nonempty_rate": len(nonempty) / max(1, len(generated)),
        "unique_ratio": len({norm for _, norm in nonempty}) / denominator,
        "exact_match_rate": sum(norm in reference_set for _, norm in nonempty) / denominator,
        "training_caption_coverage": len(exact_values) / max(1, len(reference_set)),
        "mean_nearest_similarity": sum(similarities) / max(1, len(similarities)),
        "max_nearest_similarity": max(similarities, default=0.0),
        "mean_words": sum(word_counts) / max(1, len(word_counts)),
        "repeated_bigram_rate": sum(has_repeated_bigram(raw) for raw, _ in nonempty) / denominator,
        "most_common_generations": Counter(norm for _, norm in nonempty).most_common(10),
        "closest_examples": sorted(
            closest, key=lambda row: row["similarity"], reverse=True
        )[:10],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_paths(args: argparse.Namespace) -> None:
    args.checkpoint_root = args.checkpoint_root or args.experiment_root / "train"
    args.data = args.data or args.experiment_root / "data" / "train"
    args.references = args.references or args.experiment_root / "data" / "references.jsonl"
    saved_config = args.checkpoint_root / "config.yml"
    args.config = args.config or (
        saved_config if saved_config.is_file() else Path("configs/train_caption_overfit_100_ELF-B.yml")
    )


def validate_args(args: argparse.Namespace, checkpoint_steps: Iterable[int]) -> None:
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("num-samples/batch-size must be positive and num-workers non-negative")
    if args.sampling_steps < 2:
        raise ValueError("--sampling-steps must be at least 2")
    required = [args.data, args.references, args.config]
    required.extend(
        args.checkpoint_root / f"checkpoint_{step}" for step in checkpoint_steps
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing diagnostic inputs:\n  " + "\n  ".join(missing))


def read_references(path: Path) -> list[str]:
    references: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                caption = json.loads(line).get("caption")
                if isinstance(caption, str) and caption.strip():
                    references.append(caption.strip())
    if not references:
        raise ValueError(f"No captions found in {path}")
    return references


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for field in ("params", "step", "epoch"):
        if field not in checkpoint:
            raise ValueError(f"{path} is missing checkpoint field {field!r}")
    return checkpoint


def build_models(config, tokenizer, device: torch.device):
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=len(tokenizer),
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=False,
    ).to(device).eval()
    return encoder, model


def load_weight_variant(model, checkpoint: dict[str, Any], variant: str) -> None:
    if variant == "raw":
        parameters = checkpoint["params"]
    elif variant == "ema":
        parameters = checkpoint.get("ema_params1")
        if not parameters:
            raise ValueError("Checkpoint has no ema_params1")
    else:
        raise ValueError(f"Unknown parameter variant: {variant}")
    model.load_state_dict(parameters, strict=True)
    model.eval()


def first_eos_positions(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    eos = ids.eq(eos_id)
    positions = eos.to(torch.int64).argmax(dim=1)
    return torch.where(eos.any(dim=1), positions, torch.full_like(positions, -1))


def batch_value_to_tensor(
    value: Any, device: torch.device, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Move either a NumPy collate value or an existing tensor to a device."""
    tensor = torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


@torch.inference_mode()
def run_reconstruction_variant(
    *, model, encoder, tokenizer, dataset, config, device, batch_size, num_workers,
    references, output_path, tqdm_class,
) -> dict[str, Any]:
    pad_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token for reconstruction diagnostics")
    loader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_id,
        max_input_seq_length=config.max_input_length,
        distributed=False,
    )
    records: list[dict[str, Any]] = []
    correct_tokens = valid_tokens = exact_tokens = eos_emitted = eos_position_correct = 0
    offset = 0
    model_dtype = next(model.parameters()).dtype
    for batch in tqdm_class(loader, desc=f"Reconstruct {output_path.parent.name}", unit="batches"):
        # get_dataloader's custom collate function intentionally returns NumPy
        # arrays. Training calls prepare_batch before device transfer; this
        # standalone diagnostic performs the equivalent conversion here.
        target_ids = batch_value_to_tensor(
            batch["input_ids"], device, dtype=torch.long
        )
        attention = batch_value_to_tensor(
            batch["attention_mask"], device, dtype=torch.float32
        )
        encoder_attention = batch_value_to_tensor(
            batch["encoder_attention_mask"], device, dtype=torch.float32
        )
        latent = encode_text(
            target_ids,
            encoder_attention,
            encoder,
            config.latent_mean,
            config.latent_std,
            use_bf16=config.use_bf16,
        ).to(model_dtype)
        predicted = _dlm_decode_batch(latent, model, 1.0, config, 1.0)
        valid = attention.bool()
        correct_tokens += int((predicted.eq(target_ids) & valid).sum().item())
        valid_tokens += int(valid.sum().item())
        exact_batch = (predicted.eq(target_ids) | ~valid).all(dim=1)
        exact_tokens += int(exact_batch.sum().item())
        predicted_eos = first_eos_positions(predicted, eos_id)
        target_eos = first_eos_positions(target_ids, eos_id)
        eos_emitted += int(predicted_eos.ge(0).sum().item())
        eos_position_correct += int(predicted_eos.eq(target_eos).sum().item())
        display_ids = mask_after_eos(predicted, eos_id, pad_id)
        for row in range(target_ids.shape[0]):
            generated = tokenizer.decode(display_ids[row].cpu().tolist(), skip_special_tokens=True)
            target = tokenizer.decode(target_ids[row].cpu().tolist(), skip_special_tokens=True)
            records.append(
                {
                    "id": offset + row,
                    "target": target,
                    "generated": generated,
                    "exact_token_match": bool(exact_batch[row].item()),
                    "predicted_eos_position": int(predicted_eos[row].item()),
                    "target_eos_position": int(target_eos[row].item()),
                }
            )
        offset += target_ids.shape[0]
    write_jsonl(output_path, records)
    generated = [record["generated"] for record in records]
    metrics = generation_metrics(generated, references)
    metrics.update(
        {
            "num_examples": len(records),
            "token_accuracy": correct_tokens / max(1, valid_tokens),
            "exact_token_reconstruction_rate": exact_tokens / max(1, len(records)),
            "exact_text_reconstruction_rate": sum(
                normalize_text(record["generated"]) == normalize_text(record["target"])
                for record in records
            ) / max(1, len(records)),
            "eos_emission_rate": eos_emitted / max(1, len(records)),
            "eos_position_accuracy": eos_position_correct / max(1, len(records)),
        }
    )
    return metrics


def reset_random_seed(seed: int, device: torch.device) -> torch.Generator:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


@torch.inference_mode()
def generate_samples(
    *, model, tokenizer, config, sampling_config, num_samples, batch_size,
    seed, device, tqdm_class,
) -> list[str]:
    pad_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token for generation diagnostics")
    model_dtype = next(model.parameters()).dtype
    generated: list[str] = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    for batch_index in tqdm_class(range(num_batches), desc="Generate", unit="batches"):
        current_batch = min(batch_size, num_samples - len(generated))
        generator = reset_random_seed(seed + batch_index, device)
        z = torch.randn(
            (current_batch, config.max_length, model.text_encoder_dim),
            dtype=model_dtype,
            device=device,
        ) * config.denoiser_noise_scale
        t_steps = get_sampling_steps(
            n_steps=sampling_config.num_sampling_steps[0],
            time_schedule=sampling_config.time_schedule,
            P_mean=config.denoiser_p_mean,
            P_std=config.denoiser_p_std,
            device=device,
            dtype=model_dtype,
        )
        latent = _generate_samples_single_batch(
            model=model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling_config,
            cfg_scale=1.0,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        predicted = _dlm_decode_batch(
            latent,
            model,
            float(t_steps[-1].item()),
            config,
            sampling_config.self_cond_cfg_scales[0],
        )
        predicted = mask_after_eos(predicted, eos_id, pad_id)
        generated.extend(
            tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
            for row in predicted
        )
    return generated[:num_samples]


def sampling_config(method: str, scale: float, args: argparse.Namespace) -> SamplingConfig:
    return SamplingConfig(
        sampling_method=method,
        num_sampling_steps=[args.sampling_steps],
        cfgs=[1],
        self_cond_cfg_scales=[scale],
        time_schedule="logit_normal",
        sde_gamma=args.sde_gamma if method == "sde" else 0.0,
    )


def generation_run(
    *, model, tokenizer, config, sc, args, references, output_path, tqdm_class
) -> dict[str, Any]:
    generated = generate_samples(
        model=model,
        tokenizer=tokenizer,
        config=config,
        sampling_config=sc,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=next(model.parameters()).device,
        tqdm_class=tqdm_class,
    )
    write_jsonl(
        output_path,
        ({"id": index, "generated": text} for index, text in enumerate(generated)),
    )
    return generation_metrics(generated, references)


def flat_metric_row(metadata: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    scalar_metrics = {
        key: value for key, value in metrics.items() if not isinstance(value, (dict, list))
    }
    return {**metadata, **scalar_metrics}


def runtime_manifest(
    args: argparse.Namespace, device: torch.device, checkpoint_steps: Iterable[int]
) -> dict[str, Any]:
    """Build a reproducibility manifest shared by the experiment outputs."""
    return {
        "device": str(device),
        "config": str(args.config),
        "data": str(args.data),
        "references": str(args.references),
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_steps": list(checkpoint_steps),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sampling_steps": args.sampling_steps,
        "sde_gamma": args.sde_gamma,
        "seed": args.seed,
    }


def prepare_runtime(args: argparse.Namespace, checkpoint_steps: Iterable[int]):
    """Resolve inputs, load the dataset/reference set, and build ELF/T5."""
    os.chdir(REPO_ROOT)
    resolve_paths(args)
    validate_args(args, checkpoint_steps)
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0"
    )
    if device.type != "cuda" and not args.use_cpu:
        raise RuntimeError("CUDA GPU is required; pass --use-cpu only for a slow smoke test")
    config = load_config_from_yaml(str(args.config))
    config.data_path = str(args.data)
    config.use_compile = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name
    )
    dataset = load_dataset_split(str(args.data))
    if args.num_samples > len(dataset):
        raise ValueError(
            f"Requested {args.num_samples} samples but dataset only has {len(dataset)}"
        )
    dataset = dataset.select(range(args.num_samples))
    references = read_references(args.references)[: args.num_samples]
    if len(references) != args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples but only found {len(references)} references"
        )
    encoder, model = build_models(config, tokenizer, device)
    return config, tokenizer, dataset, references, encoder, model, tqdm, device
