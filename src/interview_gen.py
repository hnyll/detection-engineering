"""Interview companion: generates the four skeletons with the best real
experiment's numbers and artifact paths pre-filled. Smoke tests are excluded,
dataset facts come from experiment metadata (never hardcoded), and existing
files are only overwritten with --force — they hold your edits."""

from __future__ import annotations

import json
from pathlib import Path

from .expmeta import load_config
from .paths import COCO_GT_DIR, INTERVIEW, list_experiments, resolve_exp

SMOKE_MARKERS = ("phase0", "smoke")


def _metrics(exp: Path) -> dict | None:
    mj = exp / "metrics.json"
    return json.loads(mj.read_text()) if mj.exists() else None


def _is_smoke(exp: Path, metrics: dict | None) -> bool:
    if any(m in exp.name for m in SMOKE_MARKERS):
        return True
    # judge by what was actually EVALUATED (metrics.protocol), not the current
    # config — the config can be edited after the fact
    if metrics:
        return metrics["protocol"]["dataset"] == "coco128"
    try:
        return load_config(exp)["meta"]["dataset"] == "coco128"
    except SystemExit:
        return True  # unreadable config — never build interview claims on it


def _best_exp() -> tuple[Path | None, dict | None]:
    from .expmeta import stale_seeds

    best, best_score, best_m = None, -1.0, None
    for e in list_experiments():
        m = _metrics(e)
        if m is None or _is_smoke(e, m):
            continue
        if stale_seeds(e, m):
            print(f"skipping {e.name}: metrics predate a retrained checkpoint — re-run eval")
            continue
        score = (m.get("aggregate") or {}).get("map50_95_mean") or 0
        if score > best_score:
            best, best_score, best_m = e, score, m
    return best, best_m


def _dataset_facts(metrics: dict) -> dict:
    """Facts from the dataset the metrics were actually produced on."""
    dataset = metrics["protocol"]["dataset"]
    split = metrics["protocol"]["split"]
    facts = {"dataset": dataset, "split": split, "n_images": "TODO", "n_classes": "TODO"}
    gt = COCO_GT_DIR / f"{dataset}_{split}.json"
    if gt.exists():
        g = json.loads(gt.read_text())
        facts["n_images"] = len(g["images"])
        facts["n_classes"] = len(g["categories"])
    return facts


def generate(exp_ref: str | None = None, force: bool = False) -> None:
    from .expmeta import protocol_hash

    if exp_ref:
        from .expmeta import stale_seeds

        exp = resolve_exp(exp_ref)
        metrics = _metrics(exp)
        if metrics is None:
            raise SystemExit(f"{exp.name} has no metrics.json — eval first")
        if _is_smoke(exp, metrics):
            raise SystemExit(f"{exp.name} is a smoke test — interview claims must "
                             "come from a real experiment")
        if stale_seeds(exp, metrics):
            raise SystemExit(f"{exp.name} metrics predate a retrained checkpoint — "
                             "re-run eval before citing them")
    else:
        exp, metrics = _best_exp()
    if exp is None:
        raise SystemExit("no real (non-smoke) experiment with metrics.json yet — "
                         "run a VisDrone experiment first")
    if metrics["protocol"]["hash"] != protocol_hash():
        print("WARNING: metrics predate the current protocol.yaml — numbers cited "
              "here are from the OLD protocol; consider re-running eval")

    facts = _dataset_facts(metrics)
    agg = metrics.get("aggregate") or {}
    map_s = agg.get("map50_95_mean", "TODO")
    lat = agg.get("latency_ms_mean_mean") or (metrics.get("speed") or {}).get("latency_ms_mean", "TODO")
    rel = f"experiments/{exp.name}"
    cite = (f"(evidence: {rel}/metrics.json, {rel}/tide_report.json, "
            f"{rel}/decision.md)")

    INTERVIEW.mkdir(exist_ok=True)
    files = {
        "resume_bullet.md": f"""# Resume bullet — Action + Scale + Result + Engineering Judgment

> TODO-draft: Built a controlled-experiment object detection pipeline (YOLO on
> {facts['dataset']}, {facts['n_images']} {facts['split']} images / {facts['n_classes']} classes);
> ran seeded ablations under a fixed evaluation protocol reaching mAP50-95 {map_s}
> at {lat} ms/img on an RTX 4060, diagnosed dominant failure modes with TIDE +
> FiftyOne, and shipped ONNX/TensorRT exports with documented drift checks.

Rules: one line, quantified, mentions judgment (protocol discipline / negative
results), no tool-soup. {cite}
""",
        "star.md": f"""# STAR story — {exp.name}

## Situation
TODO: the project context ({facts['dataset']} under an 8GB VRAM constraint).

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

1. **Problem** (15s): dense small-object detection on {facts['dataset']}; why it's hard.
2. **Setup** (20s): YOLO baselines under a fixed eval protocol; W&B tracking;
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

1. **Why is {facts['dataset']} hard for stock detectors?**
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
        target = INTERVIEW / name
        if target.exists() and not force:
            print(f"interview/{name} exists — keeping your edits (use --force to regenerate)")
            continue
        target.write_text(text)
        print(f"wrote interview/{name}")
    print(f"skeletons cite {rel} — replace every TODO with evidence before the gate counts it")
