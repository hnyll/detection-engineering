#!/usr/bin/env bash
# Reproduction commands for the generated label view and Exp3.
python3 scripts/build_grouped_dataset.py \
  --source-yaml configs/visdrone.yaml \
  --mapping experiments/exp_003_class_grouping/class_groups.csv \
  --output data/derived/visdrone_exp003_grouped

# Review the mapping before running these:
# make train EXP=exp_003_class_grouping
# make eval EXP=exp_003_class_grouping
# make tide EXP=exp_003_class_grouping ARGS="--seed 17"
# make review EXP=exp_003_class_grouping ARGS="--seed 17 --launch"

# 2026-08-04T14:13:22+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_003_class_grouping

# 2026-08-04T14:13:58+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_003_class_grouping

# 2026-08-04T14:14:15+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_003_class_grouping

# 2026-08-04T14:21:23+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_003_class_grouping

# 2026-08-04T14:36:22+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_003_class_grouping

# 2026-08-05T02:37:29+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_003_class_grouping

# 2026-08-05T02:40:59+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_003_class_grouping --seed 17

# 2026-08-05T02:50:44+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_003_class_grouping --seed 17 --launch

# 2026-08-05T09:20:27+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_003_class_grouping --seed 17 --launch
