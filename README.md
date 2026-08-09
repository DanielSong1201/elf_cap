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

## Reference code and papers

The sibling `../elf_torch/` checkout and `../papers/` directory are reference
materials only. They are intentionally not tracked by this repository.
