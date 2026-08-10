#!/usr/bin/env bash

# 1. Create the remapped-label view. Images are hardlinked or symlinked.
python3 scripts/build_grouped_dataset.py \
  --source-yaml configs/visdrone.yaml \
  --mapping experiments/exp_006_coco_aligned_balanced/class_groups.csv \
  --output data/derived/visdrone_exp006_coco_aligned

# 2. Create a deterministic repeat-factor training manifest over remapped labels.
python3 scripts/build_repeat_factor_manifest.py \
  --source-yaml configs/visdrone_exp006_coco_aligned.yaml \
  --output experiments/exp_006_coco_aligned_balanced/artifacts/train_repeat_factor.txt \
  --repeat-threshold 0.80 \
  --max-repeat 3 \
  --seed 17

# 3. Train and evaluate the custom six-class task.
make train EXP=exp_006_coco_aligned_balanced ARGS="--seeds 17"
make eval EXP=exp_006_coco_aligned_balanced ARGS="--seed 17"
make tide EXP=exp_006_coco_aligned_balanced ARGS="--seed 17"

# 2026-08-06T16:10:46+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_006_coco_aligned_balanced

# 2026-08-07T02:12:25+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_006_coco_aligned_balanced --seed 17

# 2026-08-07T02:13:32+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_006_coco_aligned_balanced --seed 17
