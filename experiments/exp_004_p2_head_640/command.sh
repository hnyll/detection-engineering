#!/usr/bin/env bash
# Exp4 reproduction commands (original ten-class Task 1).
# make train EXP=exp_004_p2_head_640
# make eval EXP=exp_004_p2_head_640
# make tide EXP=exp_004_p2_head_640 ARGS="--seed 17"
# make review EXP=exp_004_p2_head_640 ARGS="--seed 17 --launch"

# 2026-08-05T22:34:45+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_004_p2_head_640

# 2026-08-05T23:24:37+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_004_p2_head_640

# 2026-08-05T23:26:17+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_004_p2_head_640

# 2026-08-05T23:31:15+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli train exp_004_p2_head_640

# 2026-08-06T06:52:01+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_004_p2_head_640 --seed 17

# 2026-08-06T07:06:32+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_004_p2_head_640 --seed 17

# 2026-08-06T07:08:13+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_004_p2_head_640 --seed 17

# 2026-08-06T07:11:41+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_004_p2_head_640 --seed 17

# 2026-08-06T07:12:51+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_004_p2_head_640 --seed 17

# 2026-08-06T07:13:38+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli eval exp_004_p2_head_640 --seed 17

# 2026-08-06T07:14:41+00:00 | git e10f4e5 | torch 2.9.1+cu128 | ultralytics 8.4.65
uv run python -m src.cli tide exp_004_p2_head_640 --seed 17
