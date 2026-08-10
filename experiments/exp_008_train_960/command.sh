#!/usr/bin/env bash

# Exp8 reuses the existing six-class balanced dataset view. No dataset rebuild
# is required. Run these commands from the repository root.
make train EXP=exp_008_train_960 ARGS="--seeds 17"
make eval EXP=exp_008_train_960 ARGS="--seed 17"
make tide EXP=exp_008_train_960 ARGS="--seed 17"

# Valid point comparison: Exp7 and Exp8 share the 960 evaluation protocol.
# With one training seed, the comparison is exploratory rather than statistical.
make compare EXPS="exp_007_inference_960 exp_008_train_960"

# 2026-08-08T03:26:21+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_008_train_960

# 2026-08-08T13:50:31+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_008_train_960 --seed 17

# 2026-08-08T13:51:38+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_008_train_960 --seed 17

# 2026-08-08T13:51:40+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_007_inference_960 exp_008_train_960

# 2026-08-10T01:56:28+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_008_train_960 exp_010_class_weighted_960
