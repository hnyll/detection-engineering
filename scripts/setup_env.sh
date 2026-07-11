#!/usr/bin/env bash
# Reproducible environment bootstrap: installs uv if missing, then syncs the
# project-local .venv exactly from pyproject.toml + uv.lock.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Syncing core environment (.venv)"
uv sync

echo "==> Syncing TensorRT group (optional; Deploy competency engine benchmarks)"
uv sync --group trt || echo "WARNING: TensorRT group failed to install — TRT benchmarks unavailable; ONNX path unaffected."

echo "==> Done. Run 'make doctor' to verify."
