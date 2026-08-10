#!/usr/bin/env bash
# Reproduction commands (appended by the harness)

# 2026-08-03T14:51:37+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-03T14:53:55+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-03T14:58:32+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-03T15:02:57+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-03T15:07:13+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-04T02:07:28+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --resume

# 2026-08-04T02:15:04+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024

# 2026-08-04T07:47:48+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_002_resolution_1024

# 2026-08-04T07:53:52+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_002_resolution_1024 --seed 17

# 2026-08-04T09:13:37+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_001_baseline_640 exp_002_resolution_1024

# 2026-08-04T11:05:16+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_002_resolution_1024 --seed 17 --launch

# 2026-08-05T11:33:16+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_002_resolution_1024 --seed 17 --export-stats

# 2026-08-05T12:12:51+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42,1337

# 2026-08-05T12:13:37+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42,1337

# 2026-08-05T12:14:18+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42,1337

# 2026-08-05T12:17:14+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42,1337

# 2026-08-05T12:17:28+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42,1337

# 2026-08-05T12:25:19+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42

# 2026-08-05T12:27:09+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 42

# 2026-08-05T17:25:34+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_002_resolution_1024 --seeds 1337

# 2026-08-05T22:26:47+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_002_resolution_1024 --seed 42

# 2026-08-05T22:31:57+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_002_resolution_1024 --seed 42

# 2026-08-05T22:32:07+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_002_resolution_1024 --seed 1337

# 2026-08-05T22:34:39+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_002_resolution_1024 --seed 1337

# 2026-08-10T10:07:08+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli compare exp_001_baseline_640 exp_002_resolution_1024

# 2026-08-10T10:07:26+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_002_resolution_1024 --seed 17
