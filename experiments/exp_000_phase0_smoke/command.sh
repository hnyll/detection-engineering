#!/usr/bin/env bash
# Reproduction commands (appended by the harness)

# 2026-07-11T14:53:42+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke

# 2026-07-11T14:55:32+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke

# 2026-07-11T14:56:44+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --resume

# 2026-07-11T14:57:20+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --seeds 42

# 2026-07-11T14:59:09+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke

# 2026-07-11T14:59:51+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --resume

# 2026-07-11T15:02:03+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke

# 2026-07-11T15:02:54+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --resume

# 2026-07-11T15:04:06+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --seeds 42

# 2026-07-11T15:11:43+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke

# 2026-07-11T15:12:33+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --resume

# 2026-07-11T15:13:55+00:00 | git b5ba2e1 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_000_phase0_smoke --seeds 42

# 2026-07-11T16:00:44+00:00 | git fde1cd2 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_000_phase0_smoke

# 2026-07-12T04:32:44+00:00 | git 9b3a02e | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_000_phase0_smoke

# 2026-07-12T05:28:31+00:00 | git ee19924 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_000_phase0_smoke

# 2026-07-12T05:28:54+00:00 | git ee19924 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_000_phase0_smoke

# 2026-07-12T05:28:59+00:00 | git ee19924 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli review exp_000_phase0_smoke --seed 17 --export-stats

# 2026-07-12T05:31:00+00:00 | git ee19924 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli parity exp_000_phase0_smoke --seed 17

# 2026-07-12T05:31:00+00:00 | git ee19924 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli export exp_000_phase0_smoke --format onnx --seed 17
