#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-outputs/experiments/elf_caption_overfit_100_ema099}
DATA_DIR=${DATA_DIR:-${EXPERIMENT_ROOT}/data}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-${EXPERIMENT_ROOT}/train}
ANALYSIS_DIR=${ANALYSIS_DIR:-${EXPERIMENT_ROOT}/analysis}
SOURCE_DATA=${SOURCE_DATA:-outputs/processed/audiocaps_caption_only/hf_dataset/train}
CONFIG=${CONFIG:-configs/train_caption_overfit_100_ELF-B.yml}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-100}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-10}
EMA_DECAY=${EMA_DECAY:-0.99}

export PYTHONPATH="${REPO_ROOT}/elf${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

for value_name in EXPERIMENT_ROOT DATA_DIR TRAIN_OUTPUT ANALYSIS_DIR SOURCE_DATA CONFIG; do
    value=${!value_name}
    if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
        echo "ERROR: ${value_name} contains a newline; check the shell assignment." >&2
        exit 2
    fi
done

echo "Experiment root:  ${EXPERIMENT_ROOT}"
echo "Training data:    ${DATA_DIR}/train"
echo "Training output:  ${TRAIN_OUTPUT}"
echo "EMA decay:        ${EMA_DECAY}"
echo "Epochs/batch:     ${EPOCHS}/${GLOBAL_BATCH_SIZE}"
echo "DataLoader workers: ${NUM_WORKERS}"

gpu_check=(
    python -c
    'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__)'
)
"${gpu_check[@]}"

subset_cmd=(
    python scripts/build_caption_overfit_subset.py
    --input-dataset "${SOURCE_DATA}"
    --output-dir "${DATA_DIR}"
    --num-samples 100
    --seed 42
    --num-workers "${NUM_WORKERS}"
)

if [[ "${REBUILD_DATA:-0}" == "1" ]]; then
    "${subset_cmd[@]}" --overwrite
elif [[ ! -f "${DATA_DIR}/train/dataset_info.json" ]]; then
    "${subset_cmd[@]}"
else
    echo "Using existing deterministic subset: ${DATA_DIR}/train"
fi

if [[ ! -f "${DATA_DIR}/train/dataset_info.json" ]]; then
    echo "ERROR: overfit Arrow dataset was not created: ${DATA_DIR}/train" >&2
    exit 1
fi
if [[ ! -f "${DATA_DIR}/references.jsonl" ]]; then
    echo "ERROR: overfit references were not created: ${DATA_DIR}/references.jsonl" >&2
    exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}" "${TRAIN_OUTPUT}"

train_cmd=(
    python -u elf/train.py
    --config "${CONFIG}"
    --config_override "data_path=${DATA_DIR}/train"
    --config_override "output_dir=${TRAIN_OUTPUT}"
    --config_override "epochs=${EPOCHS}"
    --config_override "global_batch_size=${GLOBAL_BATCH_SIZE}"
    --config_override "num_workers=${NUM_WORKERS}"
    --config_override "ema_decay1=${EMA_DECAY}"
)
"${train_cmd[@]}" 2>&1 | tee -a "${EXPERIMENT_ROOT}/train.log"

analysis_cmd=(
    python scripts/analyze_caption_overfit.py
    --references "${DATA_DIR}/references.jsonl"
    --generation-root "${TRAIN_OUTPUT}"
    --train-log "${EXPERIMENT_ROOT}/train.log"
    --output-dir "${ANALYSIS_DIR}"
)
"${analysis_cmd[@]}"

echo "Experiment complete."
echo "Training output: ${TRAIN_OUTPUT}"
echo "Analysis report: ${ANALYSIS_DIR}/analysis.json"
