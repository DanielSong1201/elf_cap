#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-outputs/experiments/elf_caption_overfit_100}
DATA_DIR=${DATA_DIR:-${EXPERIMENT_ROOT}/data}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${EXPERIMENT_ROOT}/train}
ANALYSIS_DIR=${ANALYSIS_DIR:-${EXPERIMENT_ROOT}/analysis}
SOURCE_DATA=${SOURCE_DATA:-outputs/processed/audiocaps_caption_only/hf_dataset/train}
CONFIG=${CONFIG:-configs/train_caption_overfit_100_ELF-B.yml}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-10}

export PYTHONPATH="${REPO_ROOT}/elf${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__)'

if [[ "${REBUILD_DATA:-0}" == "1" ]]; then
    python scripts/build_caption_overfit_subset.py \
        --input-dataset "${SOURCE_DATA}" \
        --output-dir "${DATA_DIR}" \
        --num-samples 100 \
        --seed 42 \
        --num-workers "${NUM_WORKERS}" \
        --overwrite
elif [[ ! -f "${DATA_DIR}/train/dataset_info.json" ]]; then
    python scripts/build_caption_overfit_subset.py \
        --input-dataset "${SOURCE_DATA}" \
        --output-dir "${DATA_DIR}" \
        --num-samples 100 \
        --seed 42 \
        --num-workers "${NUM_WORKERS}"
else
    echo "Using existing deterministic subset: ${DATA_DIR}/train"
fi

mkdir -p "${EXPERIMENT_ROOT}" "${TRAIN_OUTPUT}"

python -u elf/train.py \
    --config "${CONFIG}" \
    --config_override "data_path=${DATA_DIR}/train" \
    --config_override "output_dir=${TRAIN_OUTPUT}" \
    --config_override "epochs=${EPOCHS}" \
    --config_override "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    --config_override "num_workers=${NUM_WORKERS}" \
    2>&1 | tee -a "${EXPERIMENT_ROOT}/train.log"

python scripts/analyze_caption_overfit.py \
    --references "${DATA_DIR}/references.jsonl" \
    --generation-root "${TRAIN_OUTPUT}" \
    --train-log "${EXPERIMENT_ROOT}/train.log" \
    --output-dir "${ANALYSIS_DIR}"

echo "Experiment complete."
echo "Training output: ${TRAIN_OUTPUT}"
echo "Analysis report: ${ANALYSIS_DIR}/analysis.json"
