#!/usr/bin/env bash

# Build the deterministic training manifest before training.
uv run python scripts/build_repeat_factor_manifest.py \
  --source-yaml configs/visdrone.yaml \
  --output experiments/exp_005_class_balanced_sampling/artifacts/train_repeat_factor.txt \
  --repeat-threshold 0.80 \
  --max-repeat 3 \
  --seed 17

make train EXP=exp_005_class_balanced_sampling ARGS="--seeds 17"
make eval EXP=exp_005_class_balanced_sampling ARGS="--seed 17"
make tide EXP=exp_005_class_balanced_sampling ARGS="--seed 17"

# 2026-08-06T09:24:33+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_005_class_balanced_sampling

# 2026-08-06T15:54:14+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_005_class_balanced_sampling --seed 17

# 2026-08-06T15:55:15+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_005_class_balanced_sampling --seed 17
