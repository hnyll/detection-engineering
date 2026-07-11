"""Competency gates: mechanical ✓/✗ against the spec's exit criteria.

Only checks what a filesystem/git read can prove; judgment calls (e.g. whether a
failure analysis is *good*) stay human. A check passing here is necessary, not
sufficient."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .paths import COMPARISONS, INTERVIEW, ROOT, TEMPLATES, list_experiments


def _state(exp: Path) -> dict:
    p = exp / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"runs": []}


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _frontmatter(exp: Path) -> dict:
    from .expmeta import read_frontmatter
    return read_frontmatter(exp / "decision.md")


def _differs_from_template(exp: Path, fname: str) -> bool:
    f = exp / fname
    if not f.exists():
        return False
    tpl = (TEMPLATES / fname).read_text().replace("{EXP}", exp.name)
    return f.read_text().strip() != tpl.strip()


def _git_tracked(pattern: str) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        return [l for l in out.splitlines() if l]
    except Exception:
        return []


def _names(items, n=3) -> str:
    labels = []
    for x in items[:n]:
        if isinstance(x, tuple):
            x = x[0]
        labels.append(getattr(x, "name", None) or str(x))
    return ", ".join(labels) if labels else ""


def _checks() -> dict[str, list[tuple[str, bool, str]]]:
    exps = list_experiments()
    runs = [(e, r) for e in exps for r in _state(e)["runs"]]
    completed = [(e, r) for e, r in runs if r.get("status") == "completed"]

    has_metrics = [e for e in exps if (e / "metrics.json").exists()]
    resumed = [(e, r) for e, r in completed if r.get("resumed_from")]
    with_wandb = [(e, r) for e, r in runs if r.get("wandb_id") or r.get("wandb_runs")]

    tide_done = [e for e in exps if (e / "tide_report.json").exists()]
    reviews = {e: _json(e / "artifacts" / "fiftyone_review.json") for e in exps}
    reviewed_100 = [e for e, r in reviews.items() if r.get("reviewed_failures", 0) >= 100]
    fr_filled = [e for e in exps if _differs_from_template(e, "failure_report.md")]
    top3 = [e for e in fr_filled
            if all(re.search(rf"^{i}\.\s*\S+", (e / "failure_report.md").read_text(), re.M)
                   for i in (1, 2, 3))]

    hyp_filled = [e for e in exps if _differs_from_template(e, "hypothesis.md")]
    decided = [e for e in exps if _frontmatter(e).get("outcome") in
               ("positive", "negative", "inconclusive")]
    controlled = [e for e in exps if e in hyp_filled and e in decided]
    negatives = [e for e in decided if _frontmatter(e).get("outcome") == "negative"]

    cmps = [_json(p) for p in COMPARISONS.glob("cmp_*.json")] if COMPARISONS.exists() else []
    cmp_complete = [c for c in cmps if c.get("experiments") and all(
        row.get(k) is not None for row in c["experiments"]
        for k in ("latency_ms", "fps", "params_m", "gflops"))]
    cmp_3seed = [c for c in cmps if c.get("experiments")
                 and all(row.get("n_seeds", 0) >= 3 for row in c["experiments"])]

    parities = [e for e in exps if (e / "exports" / "parity_report.json").exists()]
    benches = {e: _json(e / "exports" / "bench.json") for e in exps}
    trt = [e for e, b in benches.items() if any(k.startswith("trt") for k in b)]
    coreml_exports = [e for e in exps if list((e / "exports").glob("*.mlpackage"))]
    coreml_manual = bool(re.search(r"\| Date / device \|\s*[^|\s]",
                                   (ROOT / "docs" / "deploy_coreml.md").read_text()))
    coreml_ok = bool(coreml_exports) or coreml_manual

    interview_files = ["resume_bullet.md", "star.md", "walkthrough.md", "qa.md"]
    iv_ok = []
    for f in interview_files:
        p = INTERVIEW / f
        if p.exists():
            text = p.read_text()
            # untouched skeletons still contain TODO markers — those don't count
            if "experiments/exp_" in text and "TODO" not in text:
                iv_ok.append(f)

    return {
        "1. Train": [
            ("baseline experiment with metrics.json", bool(has_metrics), _names(has_metrics)),
            ("experiment configs tracked in git",
             bool(_git_tracked("experiments/*/config.yaml")), ""),
            ("checkpoint resume proven (completed run with resumed_from)",
             bool(resumed), _names(resumed)),
            ("W&B run recorded", bool(with_wandb), _names(with_wandb)),
            ("second clean run (≥2 completed runs)", len(completed) >= 2,
             f"{len(completed)} completed"),
        ],
        "2. Diagnose": [
            ("TIDE report", bool(tide_done), _names(tide_done)),
            ("≥100 failures reviewed in FiftyOne (tagged)", bool(reviewed_100),
             _names(reviewed_100)),
            ("failure_report.md filled", bool(fr_filled), _names(fr_filled)),
            ("top-3 failure modes written", bool(top3), _names(top3)),
        ],
        "3. Improve": [
            ("≥3 controlled hypotheses (filled hypothesis + decision)",
             len(controlled) >= 3, f"{len(controlled)}/3"),
            ("≥1 documented negative result", bool(negatives), _names(negatives)),
        ],
        "4. Compare": [
            ("comparison under equal protocol", bool(cmps), f"{len(cmps)} comparison(s)"),
            ("latency/FPS/params/GFLOPs recorded", bool(cmp_complete), ""),
            ("important comparison has 3 seeds", bool(cmp_3seed), ""),
        ],
        "5. Deploy": [
            ("ONNX parity report", bool(parities), _names(parities)),
            ("TensorRT latency benchmarked", bool(trt), _names(trt)),
            ("Core ML export or manual profile recorded", coreml_ok, ""),
        ],
        "6. Explain": [
            ("interview files cite experiment artifacts", len(iv_ok) == 4,
             f"{len(iv_ok)}/4 files"),
            ("README present", (ROOT / "README.md").exists(), ""),
        ],
    }


def report(strict: bool = False) -> None:
    all_pass = True
    for comp, checks in _checks().items():
        ok = all(c[1] for c in checks)
        all_pass &= ok
        print(f"\n{comp}  {'PASS' if ok else '—'}")
        for label, passed, detail in checks:
            mark = "✓" if passed else "✗"
            print(f"  {mark} {label}" + (f"  [{detail}]" if detail else ""))
    print("\n" + ("ALL COMPETENCIES PASS 🎉" if all_pass
                  else "Gates are honest — unmet criteria mean the work isn't done yet."))
    if strict and not all_pass:
        raise SystemExit(1)  # enforceable in CI: `gates --strict`
