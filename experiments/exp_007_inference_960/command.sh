#!/usr/bin/env bash

# Exp7 is inference-only. It inherits Exp6 seed-17 best.pt and refuses training.
make eval EXP=exp_007_inference_960 ARGS="--seed 17"
make tide EXP=exp_007_inference_960 ARGS="--seed 17"

# Do not run `make compare` for Exp6 vs Exp7: imgsz is intentionally different.

# 2026-08-08T03:03:25+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_007_inference_960 --seed 17

# 2026-08-08T03:04:30+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_007_inference_960 --seed 17

# 2026-08-08T13:51:39+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_007_inference_960 exp_008_train_960
