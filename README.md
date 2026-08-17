# ELF Audio Captioning

Research project for replacing the final autoregressive text-generation stage
of an audio-captioning system with Embedded Language Flows (ELF).

The local machine is used for development, configuration checks, unit tests,
and small CPU smoke tests. Full training and evaluation are intended to run on
a remote NVIDIA RTX 4090 server.

## Documents

- [Implementation plan](AUDIO_CAPTION_ELF_PLAN.md)

## Dataset validation

The dataset checker is read-only and uses only the Python standard library.
Run it on the server before preprocessing or training:

```bash
python scripts/check_audiocaps.py \
  --data-root data/AudioCaps_CVSSP \
  --report outputs/data_check/audiocaps_report.json
```

It expects one CSV directly inside each of `train`, `val`, and `test`. By
default, the CSV columns are `youtube_id` and `caption`, and each audio path is
resolved as `Y<youtube_id>.wav`. Multiple captions for one YouTube ID are
allowed. A nonzero exit status indicates a blocking data error.

For a fast filename/CSV-only check, skip reading WAV headers:

```bash
python scripts/check_audiocaps.py \
  --data-root data/AudioCaps_CVSSP \
  --skip-wav-header-check
```

## Build AudioCaps manifests

The manifest builder treats the CVSSP bundle as the WAV source and skips
individual rows/audio files that cannot form a valid pair. It writes one JSON
object per audio file and aggregates all valid reference captions for that
audio. Source split `val` is written as `eval.jsonl`.

If the CSV files inside the CVSSP split directories are the intended caption
source, run:

```bash
python scripts/build_audiocaps_manifests.py \
  --audio-root data/AudioCaps_CVSSP \
  --output-dir outputs/manifests/audiocaps_cvssp
```

To use CVSSP only as the audio cache and use official AudioCaps v1 captions
(recommended for reproducing the original AudioCaps benchmark), first clone
the official metadata repository and point `--metadata-root` to its `dataset`
directory:

```bash
git clone https://github.com/cdjkim/audiocaps.git /tmp/audiocaps_metadata

python scripts/build_audiocaps_manifests.py \
  --audio-root data/AudioCaps_CVSSP \
  --metadata-root /tmp/audiocaps_metadata/dataset \
  --output-dir outputs/manifests/audiocaps_v1
```

The output directory contains:

```text
train.jsonl
eval.jsonl
test.jsonl
summary.json
```

Each JSONL record has the following form:

```json
{
  "split": "eval",
  "source_split": "val",
  "audio_id": "Yabc123",
  "youtube_id": "abc123",
  "start_time": 10,
  "audio_path": "data/AudioCaps_CVSSP/val/Yabc123.wav",
  "captions": ["A dog barks.", "A dog is barking nearby."],
  "num_captions": 2,
  "audiocap_ids": ["123", "124"]
}
```

By default, audio paths are relative to the directory from which the script is
run. Use `--absolute-audio-paths` if the manifests will only be consumed on the
same server. WAV headers are checked by default. The builder skips empty
captions, missing WAV files, invalid WAV files, duplicate captions, and IDs
whose multiple start times cannot be represented by the `Y<youtube_id>.wav`
filename. Counts and examples for every skipped category are saved in
`summary.json`.

Run the standard-library tests locally with:

```bash
python -m unittest discover -s tests -v
```

## Prepare ELF caption-only data

After building the manifests, install the CPU preprocessing dependencies in
the ELF conda environment:

```bash
pip install -r requirements-data.txt
```

Convert the default AudioCaps v1 manifests into caption-level audit JSONL
files and tokenized Hugging Face Arrow datasets:

```bash
python scripts/prepare_elf_caption_data.py \
  --manifest-dir outputs/manifests/audiocaps_v1 \
  --output-dir outputs/processed/audiocaps_caption_only \
  --tokenizer t5-small \
  --max-length 48
```

The script uses tqdm progress bars for manifest expansion, JSONL writing,
tokenization, and Arrow serialization. CPU tokenization is parallel: the
default `--num-workers 0` automatically selects up to 16 CPU processes. Set an
explicit value to match the server allocation, for example:

```bash
python scripts/prepare_elf_caption_data.py \
  --manifest-dir outputs/manifests/audiocaps_v1 \
  --output-dir outputs/processed/audiocaps_caption_only \
  --num-workers 16
```

The output layout is:

```text
outputs/processed/audiocaps_caption_only/
├── train.jsonl
├── eval.jsonl
├── test.jsonl
├── summary.json
├── tokenizer/
└── hf_dataset/
    ├── train/
    ├── eval/
    └── test/
```

Each valid reference caption becomes one example. Captions are whitespace
normalized, but not lowercased or otherwise rewritten. Empty/non-string and
within-audio duplicate captions are skipped and reported in `summary.json`.
The tokenized datasets contain variable-length `input_ids` with T5 EOS and no
padding; ELF pads dynamically in its dataloader. The upstream fixed-length
OpenWebText recipe does not require special tokens, but this project adds EOS
deliberately so the short-caption adaptation can learn where generation ends.

The resulting paths can be used in an ELF training config as:

```yaml
data_path: outputs/processed/audiocaps_caption_only/hf_dataset/train
eval_data_path: outputs/processed/audiocaps_caption_only/hf_dataset/eval
encoder_model_name: t5-small
max_length: 48
```

The tokenizer must match `encoder_model_name` and the ELF checkpoint. The
script refuses to replace a non-empty output directory unless `--overwrite`
is explicitly supplied. Use `--skip-tokenization` only for a lightweight
caption-expansion check; that mode does not produce ELF-ready Arrow data.

## Reference code and papers

The sibling `../elf_torch/` checkout and `../papers/` directory are reference
materials only. They are intentionally not tracked by this repository.

## ELF-B 100-caption overfit experiment

The ELF PyTorch training implementation required by this experiment is
vendored under `elf/`; no sibling checkout, submodule, or runtime clone is
required. Its provenance and local changes are documented in
`elf/UPSTREAM.md`, and the upstream MIT license is retained in `elf/LICENSE`.

Install the server training environment:

```bash
conda create -n elf-cap python=3.10 -y
conda activate elf-cap
pip install -r requirements-train.txt
```

The experiment uses the ELF-B OpenWebText checkpoint only to initialize model
parameters. It intentionally resets optimizer, scheduler, RNG counters, step,
and epoch before AudioCaps domain adaptation. Run the complete workflow on one
4090 with:

```bash
bash scripts/run_caption_overfit_100.sh
```

The workflow:

1. deterministically selects 100 captions from the prepared training Arrow
   dataset with seed 42;
2. saves the exact selected references and source indices;
3. initializes local ELF-B from
   `embedded-language-flows/ELF-B-owt-torch`;
4. trains for 100 epochs (1,000 steps at batch size 10) with tqdm progress;
5. saves checkpoints and generates 100 captions every 20 epochs;
6. captures the training log and analyzes all generated checkpoints against
   the 100 training captions.

The short sanity run uses `ema_decay1=0.99`. A value of `0.9999` leaves about
90.5% of the initial OpenWebText EMA after only 1,000 updates and therefore
does not evaluate the newly adapted model faithfully.

Useful server overrides include:

```bash
NUM_WORKERS=16 bash scripts/run_caption_overfit_100.sh

EPOCHS=200 GLOBAL_BATCH_SIZE=10 EMA_DECAY=0.99 bash scripts/run_caption_overfit_100.sh
```

The default outputs are:

```text
outputs/experiments/elf_caption_overfit_100_ema099/
├── data/
│   ├── train/
│   ├── references.jsonl
│   └── summary.json
├── train/
│   ├── checkpoint_*
│   └── sde-*/all_generated_*.jsonl
├── train.log
└── analysis/
    ├── analysis.json
    ├── metrics.csv
    └── loss_curve.csv
```

Rerunning with existing checkpoints uses ELF's normal auto-resume behavior.
Set `REBUILD_DATA=1` only when the deterministic 100-caption subset should be
rebuilt. Use a different `EXPERIMENT_ROOT` for an independent run rather than
mixing checkpoints:

```bash
EXPERIMENT_ROOT=outputs/experiments/elf_caption_overfit_100_seed2 \
  bash scripts/run_caption_overfit_100.sh
```

The launcher builds every command as a Bash argument array. Do not copy the
rendered multi-line Python invocation out of the script: a missing continuation
backslash would make Bash execute a path fragment such as `data/train` as a
separate command. Use the launcher and environment variables above instead.

The lexical analysis reports non-empty rate, unique ratio, exact training-text
match rate, training-caption coverage, nearest-reference similarity, length,
and repeated-bigram rate. These are diagnostics for a small-data overfit test,
not final Audio Captioning metrics such as CIDEr, SPICE, SPIDEr, or FENSE.

### Diagnose the 100-caption generation failure

After the EMA-0.99 overfit run has produced `checkpoint_400` and
`checkpoint_1000`, the three experiments can be run independently:

```bash
python scripts/diagnose_latent_reconstruction.py
python scripts/diagnose_raw_vs_ema.py
python scripts/diagnose_sampler_sweep.py
```

Or run the three independent scripts sequentially with:

```bash
bash scripts/run_caption_diagnostics.sh
```

The launcher uses the existing overfit data and checkpoints and writes each
experiment separately under:

```text
outputs/diagnostics/elf_caption_overfit_100_ema099/
├── 01_latent_reconstruction/
│   ├── raw/reconstructions.jsonl
│   ├── ema/reconstructions.jsonl
│   ├── metrics.csv
│   ├── results.json
│   └── run_manifest.json
├── 02_raw_vs_ema/
│   ├── checkpoint_400/{raw,ema}/generations.jsonl
│   ├── checkpoint_1000/{raw,ema}/generations.jsonl
│   ├── metrics.csv
│   ├── results.json
│   └── run_manifest.json
└── 03_sampler_sweep/
│   ├── checkpoint_400/{ode_*,sde_*}/generations.jsonl
│   ├── checkpoint_1000/{ode_*,sde_*}/generations.jsonl
│   ├── metrics.csv
│   ├── results.json
│   └── run_manifest.json
```

The first experiment decodes clean T5 latents and reports token accuracy,
exact reconstruction, EOS emission, and EOS-position accuracy. The second
uses identical random seeds to compare raw and EMA parameters under the
original SDE/SC-CFG=3 sampler. The third uses EMA parameters to compare
ODE/SDE crossed with SC-CFG values 1 and 3.

To reduce the GPU smoke-test size for one diagnostic:

```bash
PYTHONPATH="$PWD/elf" python scripts/diagnose_latent_reconstruction.py \
  --num-samples 10 \
  --batch-size 5
```

The full launcher accepts `EXPERIMENT_ROOT`, `DIAGNOSTIC_OUTPUT`,
`BATCH_SIZE`, `NUM_WORKERS`, `NUM_SAMPLES`, and `SAMPLING_STEPS` environment
overrides. Existing result files for the same variant are replaced atomically;
unrelated output directories are not removed.

### Diagnose flow recovery from controlled noise

The next standalone diagnostic starts from a known mixture of a clean caption
latent and Gaussian noise,

```text
z_t = t * x0 + (1 - t) * noise * denoiser_noise_scale,
```

then integrates the ELF flow from `t` to 1. It compares direct decoding of
`z_t` with decoding after flow recovery, so decoder robustness is separated
from flow recovery quality. Run the default checkpoint-1000/raw/ODE sweep with:

```bash
python scripts/diagnose_flow_recovery.py
```

The default known-signal fractions are `0, 0.25, 0.5, 0.75, 0.9, 1.0`.
The `t=0` row is ordinary generation from pure noise; `t=1` is the clean
latent baseline. Each level reports valid-token latent MSE, cosine similarity,
relative L2 error both before and after flow, clean/recovered latent
distribution statistics, token/text reconstruction, EOS behavior, repetition,
and nearest-reference similarity. Per-caption comparisons are saved under:

```text
outputs/diagnostics/elf_caption_overfit_100_ema099/04_flow_recovery/
├── checkpoint_1000/raw/t_*/recoveries.jsonl
├── metrics.csv
├── results.json
└── run_manifest.json
```

Useful focused comparisons include:

```bash
# Quick ten-caption server smoke test.
python scripts/diagnose_flow_recovery.py \
  --num-samples 10 \
  --batch-size 5

# Compare training progress while keeping the primary raw parameters.
python scripts/diagnose_flow_recovery.py \
  --checkpoint-steps 400 1000

# Explicitly compare raw and EMA recovery at checkpoint 1000.
python scripts/diagnose_flow_recovery.py \
  --variants raw ema
```

The controlled recovery experiment is intentionally not part of
`run_caption_diagnostics.sh`, because its multi-level flow integrations are
substantially more expensive than the original three diagnostics.
