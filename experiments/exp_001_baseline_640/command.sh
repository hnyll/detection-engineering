#!/usr/bin/env bash
# Reproduction commands (appended by the harness)

# 2026-07-12T06:22:14+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_001_baseline_640 --seeds 17

# 2026-07-12T09:16:56+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_001_baseline_640 --seeds 42

# 2026-07-12T11:54:09+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_001_baseline_640 --seeds 1337

# 2026-07-12T14:27:27+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_001_baseline_640

# 2026-07-12T14:38:26+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_001_baseline_640

# 2026-07-12T14:38:33+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 1337 --export-stats

# 2026-07-12T14:57:57+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 1337 --export-stats

# 2026-07-12T14:59:33+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 1337 --export-stats

# 2026-07-12T15:27:01+00:00 | git 83fa8a0 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --export-stats

# 2026-07-21T12:02:01+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --launch

# 2026-07-22T12:33:29+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --launch

# 2026-07-29T13:23:54+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_001_baseline_640 --seed 17

# 2026-07-29T13:23:53+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --launch

# 2026-07-29T13:27:37+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --launch

# 2026-08-03T12:09:28+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --launch

# 2026-08-03T13:03:43+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --export-stats

# 2026-08-03T13:06:06+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_001_baseline_640 --seed 17 --export-stats

# 2026-08-04T09:13:34+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_001_baseline_640 exp_002_resolution_1024

# 2026-08-10T10:07:03+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_001_baseline_640 exp_002_resolution_1024
