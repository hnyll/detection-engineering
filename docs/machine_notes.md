# Machine notes — WSL2 / RTX 4060

Hardware: RTX 4060 (8 GB VRAM), 11 GB WSL2 RAM, 8 CPU cores, ~800 GB disk.

## The one hard rule: never touch swap
A previous session froze the machine when RAM pressure pushed WSL2 into swap-thrash.
Keep any training/eval process RSS well under ~7 GB:

- `workers=2` max in dataloaders (0–2; the data is on fast local disk).
- `cache=False` — never RAM-cache datasets.
- Explicit `batch` (16 at 640px for n/s models). **Never `batch=-1` auto** on 8 GB.
- AMP on (`amp=True`, Ultralytics default).
- High-resolution runs (1024/1280px): drop batch to 4–8 and record it as a
  compensating change in hypothesis.md.
- n/s model sizes only under the 8 GB budget.
- The train wrapper logs peak RSS per run into `state.json` — check it.

## Toolchain quirks
- No system CUDA toolkit / nvcc. Torch wheels bundle the cu128 runtime; TensorRT
  comes from pip wheels (`tensorrt-cu12`). Nothing that compiles CUDA will work.
- `unzip` is not installed — use Python `zipfile` (Ultralytics' downloader already does).
- onnxruntime-gpu finds cuDNN through torch's bundled `nvidia-cudnn-cu12` —
  always `import torch` **before** `import onnxruntime` (doctor checks this).

## FiftyOne on WSL2
- FiftyOne bundles its own `mongod` (package `fiftyone-db`); data lives in `~/.fiftyone`.
- If mongod fails to boot (rare WSL2 issue), point `FIFTYONE_DATABASE_URI` at an
  external MongoDB and retry `make doctor`.
- The app serves on `http://localhost:5151` — WSL2 forwards localhost, so open it
  in the Windows browser directly.

## W&B
- Preferred: `uv run wandb login` once (key lands in `~/.netrc`, outside the repo).
- With no credentials the harness sets `WANDB_MODE=offline`; sync later with
  `uv run wandb sync wandb/offline-run-*`.
