#!/usr/bin/env bash
# Inference-only: Exp12 inherits Exp10 seed-17 best.pt and adds the full-frame
# view to Exp11's tile candidate pool before the same global merge.
make eval EXP=exp_012_hybrid_fullframe_tiles_640 ARGS="--seed 17"
make tide EXP=exp_012_hybrid_fullframe_tiles_640 ARGS="--seed 17"

# Ordinary compare must refuse because Exp11 and Exp12 intentionally have
# distinct inference-pipeline protocol identities. Use decision.md's paired
# table instead.

# 2026-08-10T05:36:18+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_012_hybrid_fullframe_tiles_640 --seed 17

# 2026-08-10T05:39:23+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli threshold-sweep exp_012_hybrid_fullframe_tiles_640 --seed 17 --iou 0.5 --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5

# 2026-08-10T05:39:36+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_012_hybrid_fullframe_tiles_640 --seed 17
