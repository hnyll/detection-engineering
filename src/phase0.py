"""Phase 0 — COCO128 plumbing smoke test (spec: prove the loop, draw no
modeling conclusions). One command exercises: train -> planned interrupt ->
checkpoint resume -> second clean run -> fixed-protocol eval -> TIDE ->
FiftyOne -> ONNX export + parity -> gates snapshot."""

from __future__ import annotations

import yaml

from . import paths
from .scaffold import new_experiment

SMOKE_NAME = "phase0_smoke"

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
        "seeds": [17],
        "causal_variable": "none — plumbing smoke test",
        "compensating_changes": [],
    },
    "train": {"model": "yolo11n.pt", "imgsz": 640, "epochs": 3, "batch": 16},
}


def _ensure_exp():
    for d in paths.list_experiments():
        if d.name.endswith(f"_{SMOKE_NAME}"):
            return d
    new_experiment(SMOKE_NAME)
    exp = [d for d in paths.list_experiments() if d.name.endswith(f"_{SMOKE_NAME}")][0]
    (exp / "config.yaml").write_text(yaml.safe_dump(CONFIG, sort_keys=False))
    (exp / "hypothesis.md").write_text(HYPOTHESIS.format(exp=exp.name))
    return exp


def run_phase0() -> None:
    from .evaluate import evaluate
    from .export import export_model, parity
    from .fo_review import review
    from .gates import report
    from .tide_wrap import run_tide
    from .train import train

    exp = _ensure_exp()
    name = exp.name
    steps = [
        ("train (planned interrupt after epoch 1)",
         lambda: train(name, seeds=[17], interrupt_after=1)),
        ("resume from checkpoint", lambda: train(name, seeds=[17], resume=True)),
        ("second clean run (seed 42)", lambda: train(name, seeds=[42])),
        ("fixed-protocol eval", lambda: evaluate(name)),
        ("TIDE error taxonomy", lambda: run_tide(name)),
        ("FiftyOne load + evaluate (headless)",
         lambda: review(name, export_stats=True)),
        ("ONNX export", lambda: export_model(name, fmt="onnx")),
        ("pt vs ONNX parity", lambda: parity(name)),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n===== phase0 [{i}/{len(steps)}] {label} =====")
        fn()

    print("\n===== phase0: gates snapshot =====")
    report()
    print("\nphase0 complete — the loop works. Phase 1 (VisDrone baseline) is next:")
    print("  make new NAME=baseline_640 && edit hypothesis.md, then make train EXP=...")
