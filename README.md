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
padding; ELF pads dynamically in its dataloader.

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
