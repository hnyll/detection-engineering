#!/usr/bin/env bash
# Inference-only: this experiment inherits Exp10 seed-17 best.pt.
make eval EXP=exp_011_tiled_inference_640 ARGS="--seed 17"
make tide EXP=exp_011_tiled_inference_640 ARGS="--seed 17"

# Ordinary compare must refuse Exp10 vs Exp11 because the inference-pipeline
# protocol identity intentionally differs. Use the paired table in decision.md.

# 2026-08-10T04:24:52+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:26:41+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:28:32+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:30:19+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:30:56+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_011_tiled_inference_640 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5

# 2026-08-10T04:41:48+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:42:02+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:43:10+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:44:59+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_011_tiled_inference_640 --seed 17

# 2026-08-10T04:45:06+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_011_tiled_inference_640 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5

# 2026-08-10T05:28:05+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T05:29:59+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_011_tiled_inference_640 --seed 17

# 2026-08-10T05:30:05+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_011_tiled_inference_640 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5

# 2026-08-10T05:33:45+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_011_tiled_inference_640 --seed 17

# 2026-08-10T05:39:23+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_011_tiled_inference_640 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5

# 2026-08-10T05:39:36+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_011_tiled_inference_640 --seed 17
