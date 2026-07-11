"""Interview companion: generates the four skeletons with the best experiment's
real numbers and artifact paths pre-filled. Every claim must point at an
artifact — the skeletons enforce that with explicit citation slots."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import INTERVIEW, ROOT, list_experiments, resolve_exp


def _best_exp() -> tuple[Path, dict] | tuple[None, None]:
    best, best_m = None, None
    for e in list_experiments():
        mj = e / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text())
        score = (m.get("aggregate") or {}).get("map50_95_mean") or 0
        if best_m is None or score > (best_m.get("aggregate") or {}).get("map50_95_mean", 0):
            best, best_m = e, m
    return best, best_m


def generate(exp_ref: str | None = None) -> None:
    if exp_ref:
        exp = resolve_exp(exp_ref)
        metrics = json.loads((exp / "metrics.json").read_text()) if (exp / "metrics.json").exists() else {}
    else:
        exp, metrics = _best_exp()
    if exp is None:
        raise SystemExit("no experiment with metrics.json yet — eval something first")

    agg = metrics.get("aggregate") or {}
    map_s = agg.get("map50_95_mean", "TODO")
    lat = agg.get("latency_ms_mean_mean") or (metrics.get("speed") or {}).get("latency_ms_mean", "TODO")
    rel = f"experiments/{exp.name}"
    cite = (f"(evidence: {rel}/metrics.json, {rel}/tide_report.json, "
            f"{rel}/decision.md)")

    INTERVIEW.mkdir(exist_ok=True)
    files = {
        "resume_bullet.md": f"""# Resume bullet — Action + Scale + Result + Engineering Judgment

> TODO-draft: Built a controlled-experiment object detection pipeline (YOLO11 on
> VisDrone-DET, 6.5k images / 10 classes); ran seeded ablations with fixed
> evaluation protocol reaching mAP50-95 {map_s} at {lat} ms/img on an RTX 4060,
> diagnosed dominant failure modes with TIDE + FiftyOne, and shipped
> ONNX/TensorRT exports with documented numerical-drift checks.

Rules: one line, quantified, mentions judgment (protocol discipline / negative
results), no tool-soup. {cite}
""",
        "star.md": f"""# STAR story — {exp.name}

## Situation
TODO: the project context (learning harness on VisDrone, 8GB VRAM constraint).

## Task
TODO: what you set out to prove/improve. Cite the pre-registered prediction in
{rel}/hypothesis.md.

## Action
TODO: the controlled experiment(s), one causal variable each, seeds, protocol.

## Result
TODO: numbers from {rel}/metrics.json (mAP {map_s}) and the decision that
followed ({rel}/decision.md). Negative results count — say what you ruled out.

{cite}
""",
        "walkthrough.md": f"""# 2-minute project walkthrough

1. **Problem** (15s): dense tiny-object detection on drone imagery; why it's hard.
2. **Setup** (20s): YOLO11 baselines under a fixed eval protocol; W&B tracking;
   8GB VRAM budget forced honest engineering tradeoffs.
3. **Diagnosis** (30s): TIDE taxonomy + FiftyOne review of ≥100 failures —
   name the top-3 failure modes from {rel}/failure_report.md.
4. **Experiments** (30s): the causal-variable ablations and what each showed;
   include the negative result.
5. **Deployment** (15s): ONNX/TensorRT parity + latency numbers from
   {rel}/exports/.
6. **What I'd do next** (10s).

{cite}
""",
        "qa.md": f"""# Likely interviewer questions — evidence-based answers

1. **Why is VisDrone hard for stock detectors?**
   TODO — cite AP_small vs AP_large gap in {rel}/metrics.json.
2. **How do you know improvement X is real and not noise?**
   TODO — 3-seed variance, improvement < std ⇒ inconclusive
   (experiments/comparisons/).
3. **Walk me through a failure you diagnosed.**
   TODO — cite {rel}/tide_report.json + a FiftyOne view.
4. **What broke when you exported the model?**
   TODO — cite {rel}/exports/parity_report.json drift numbers.
5. **What did you try that didn't work?**
   TODO — cite the experiment with decision.md `outcome: negative`.

{cite}
""",
    }
    for name, text in files.items():
        (INTERVIEW / name).write_text(text)
        print(f"wrote interview/{name}")
    print(f"skeletons cite {rel} — replace every TODO with evidence before the gate counts it")
