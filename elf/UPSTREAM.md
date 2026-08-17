# Embedded Language Flows PyTorch code

This directory vendors the training/runtime code from:

- Repository: `https://github.com/lillian039/ELF.git`
- Upstream commit: `b29d8833609e9ab7f67cd9da39435ac5cea04837`
- License: MIT; see `LICENSE` in this directory.

Local changes for this project:

1. make `elf/` directly executable as a source root inside this repository;
2. add `Config.init_checkpoint`;
3. add weight-only checkpoint initialization for domain adaptation;
4. honor `use_compile` during training instead of compiling unconditionally.

The model architecture, flow/decoder objectives, T5 encoding, sampling,
checkpoint resume behavior, and training batch construction otherwise retain
the upstream PyTorch ELF implementation.
