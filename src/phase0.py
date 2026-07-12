"""Phase 0 — COCO128 plumbing smoke test (spec: prove the loop, draw no
modeling conclusions). One command exercises: train -> planned interrupt ->
checkpoint resume -> second clean run -> fixed-protocol eval -> TIDE ->
FiftyOne -> ONNX export + parity -> gates snapshot."""

from __future__ import annotations

import urllib.request
import zipfile

import yaml

from . import paths
from .scaffold import new_experiment

SMOKE_NAME = "phase0_smoke"
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def _ensure_coco128() -> None:
    """Fetch COCO128 into data/ ourselves — relying on Ultralytics' auto-download
    extracts into the machine-global datasets_dir, not this repo."""
    if (paths.DATA / "coco128" / "images" / "train2017").exists():
        return
    paths.DATA.mkdir(exist_ok=True)
    zpath = paths.DATA / "coco128.zip"
    print(f"downloading coco128 (~7MB) -> {zpath.parent}")
    urllib.request.urlretrieve(COCO128_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(paths.DATA)
    zpath.unlink()

HYPOTHESIS = """# {exp} — Hypothesis

## Question
Does every stage of the harness (train, resume, W&B, eval, TIDE, FiftyOne,
export, parity) run end-to-end on this machine?

## Prediction (write BEFORE training)
All stages complete without manual intervention; peak RSS stays under 7GB;
resume continues from epoch 2 after the planned interrupt at epoch 1.

## Background reading / prior evidence
docs/harness_spec.md Phase 0: COCO128 proves plumbing only — 3 epochs of
yolo11n cannot support any modeling conclusion.

## Baseline
None — this is the plumbing baseline.

## Causal variable changed
None (no hypothesis about the model is being tested).

## Compensating implementation changes
None.
"""

CONFIG = {
    "meta": {
        "name": SMOKE_NAME,
        "parent_exp": None,
        "dataset": "coco128",
        # both seeds phase0 actually trains — eval/aggregation are config-driven,
        # so an undeclared seed would be excluded (and a lingering one ignored)
        "seeds": [17, 42],
        "causal_variable": "none — plumbing smoke test",
        "compensating_changes": [],
    },
    # batch 8: plumbing test only, and this box shares its 8GB card with the
    # Windows desktop — headroom beats throughput here
    "train": {"model": "yolo11n.pt", "imgsz": 640, "epochs": 3, "batch": 8},
}


def _ensure_exp():
    exp = next((d for d in paths.list_experiments()
                if d.name.endswith(f"_{SMOKE_NAME}")), None)
    if exp is None:
        new_experiment(SMOKE_NAME)
        exp = next(d for d in paths.list_experiments()
                   if d.name.endswith(f"_{SMOKE_NAME}"))
    # phase0 owns this experiment: always restore the canonical 3-epoch COCO128
    # config — a leftover template config (visdrone, 100 epochs) must never
    # masquerade as the smoke test
    current = yaml.safe_load((exp / "config.yaml").read_text()) if (exp / "config.yaml").exists() else None
    if current != CONFIG:
        (exp / "config.yaml").write_text(yaml.safe_dump(CONFIG, sort_keys=False))
        print(f"restored canonical phase0 config in {exp.name}")
    (exp / "hypothesis.md").write_text(HYPOTHESIS.format(exp=exp.name))
    return exp


def run_phase0() -> None:
    import subprocess
    import sys

    from .gates import report

    _ensure_coco128()
    exp = _ensure_exp()
    name = exp.name
    # Each GPU step runs in its own subprocess: process exit is the only reliable
    # full CUDA teardown (VRAM + WSL2 pinned-memory pool) between back-to-back
    # runs — in-process cleanup was not enough on this 8GB card.
    steps = [
        ("train (planned interrupt after epoch 1)",
         ["train", name, "--seeds", "17", "--interrupt-after", "1"]),
        ("resume from checkpoint", ["train", name, "--seeds", "17", "--resume"]),
        ("second clean run (seed 42)", ["train", name, "--seeds", "42"]),
        ("fixed-protocol eval", ["eval", name]),
        ("TIDE error taxonomy", ["tide", name]),
        ("FiftyOne load + evaluate (headless)", ["review", name, "--export-stats"]),
        ("ONNX export", ["export", name, "--format", "onnx"]),
        ("pt vs ONNX parity", ["parity", name]),
    ]
    for i, (label, args) in enumerate(steps, 1):
        print(f"\n===== phase0 [{i}/{len(steps)}] {label} =====", flush=True)
        r = subprocess.run([sys.executable, "-m", "src.cli", *args], cwd=paths.ROOT)
        if r.returncode != 0:
            raise SystemExit(f"phase0 step failed: {label} (exit {r.returncode})")

    print("\n===== phase0: gates snapshot =====")
    report()
    print("\nphase0 complete — the loop works. Phase 1 (VisDrone baseline) is next:")
    print("  make new NAME=baseline_640 && edit hypothesis.md, then make train EXP=...")
