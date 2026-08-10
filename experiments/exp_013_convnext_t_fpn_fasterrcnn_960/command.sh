#!/usr/bin/env bash
# Commands are appended by the harness when they are actually run.

# Completed real-data hardware smoke (densest unique samples first).
uv run python -m src.cli model-smoke exp_013_convnext_t_fpn_fasterrcnn_960 --batches 5 --seed 17

# 2026-08-10T07:25:30+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_013_convnext_t_fpn_fasterrcnn_960

# 2026-08-10T07:30:50+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_013_convnext_t_fpn_fasterrcnn_960
