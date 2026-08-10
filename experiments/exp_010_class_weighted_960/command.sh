#!/usr/bin/env bash

# Run from the repository root. Exp10 starts fresh from stock yolo11n.pt.
make train EXP=exp_010_class_weighted_960 ARGS="--seeds 17"
make eval EXP=exp_010_class_weighted_960 ARGS="--seed 17"
make tide EXP=exp_010_class_weighted_960 ARGS="--seed 17"
make compare EXPS="exp_008_train_960 exp_010_class_weighted_960"

# 2026-08-09T09:41:27+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_010_class_weighted_960

# 2026-08-09T13:03:01+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_010_class_weighted_960 --resume

# 2026-08-09T23:21:00+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_010_class_weighted_960 --resume

# 2026-08-10T01:55:19+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_010_class_weighted_960 --seed 17

# 2026-08-10T01:56:26+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_010_class_weighted_960 --seed 17

# 2026-08-10T01:56:29+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_008_train_960 exp_010_class_weighted_960

# 2026-08-10T04:16:10+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_010_class_weighted_960 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25

# 2026-08-10T04:17:29+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_010_class_weighted_960 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25

# 2026-08-10T04:22:36+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_010_class_weighted_960 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5
