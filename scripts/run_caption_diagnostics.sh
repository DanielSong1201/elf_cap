#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-outputs/experiments/elf_caption_overfit_100_ema099}
DIAGNOSTIC_OUTPUT=${DIAGNOSTIC_OUTPUT:-outputs/diagnostics/elf_caption_overfit_100_ema099}
BATCH_SIZE=${BATCH_SIZE:-10}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SAMPLING_STEPS=${SAMPLING_STEPS:-32}

export PYTHONPATH="${REPO_ROOT}/elf${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__)'

common_args=(
  --experiment-root "${EXPERIMENT_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-samples "${NUM_SAMPLES}"
  --sampling-steps "${SAMPLING_STEPS}"
)

python -u scripts/diagnose_latent_reconstruction.py \
  "${common_args[@]}" \
  --output-dir "${DIAGNOSTIC_OUTPUT}/01_latent_reconstruction"

python -u scripts/diagnose_raw_vs_ema.py \
  "${common_args[@]}" \
  --output-dir "${DIAGNOSTIC_OUTPUT}/02_raw_vs_ema"

python -u scripts/diagnose_sampler_sweep.py \
  "${common_args[@]}" \
  --output-dir "${DIAGNOSTIC_OUTPUT}/03_sampler_sweep"

echo "Diagnostic results: ${DIAGNOSTIC_OUTPUT}"
