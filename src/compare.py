"""Compare experiments under the identical evaluation protocol.

Refuses to compare across protocol hashes. Verdict logic per the spec's
statistical discipline: an improvement smaller than the observed seed variance
is INCONCLUSIVE; with fewer than 2 seeds on both sides the variance is unknown
and the verdict says so."""

from __future__ import annotations

import json

from . import paths
from .expmeta import protocol_hash

KEY = "map50_95"


def _load(exp_ref: str) -> tuple[str, dict]:
    exp_dir = paths.resolve_exp(exp_ref)
    mj = exp_dir / "metrics.json"
    if not mj.exists():
        raise SystemExit(f"{exp_dir.name}: no metrics.json — run eval first")
    return exp_dir.name, json.loads(mj.read_text())


def _mean_std(m: dict, key: str) -> tuple[float | None, float | None]:
    agg = m.get("aggregate", {})
    return agg.get(f"{key}_mean"), agg.get(f"{key}_std")


def compare(exp_refs: list[str]) -> None:
    exps = [_load(r) for r in exp_refs]
    if len(exps) < 2:
        raise SystemExit("need at least two experiments to compare")

    hashes = {m["protocol"]["hash"] for _, m in exps}
    if len(hashes) > 1:
        for name, m in exps:
            print(f"  {name}: protocol {m['protocol']['hash']}")
        raise SystemExit(
            "REFUSED: protocol hashes differ — results are not comparable. "
            "Re-run eval under the current configs/protocol.yaml."
        )
    if protocol_hash() not in hashes:
        raise SystemExit(
            "REFUSED: metrics were produced under an older protocol.yaml than the "
            "current one — re-run eval."
        )

    rows, verdicts = [], []
    base_name, base = exps[0]
    for name, m in exps:
        mean, std = _mean_std(m, KEY)
        rows.append({
            "exp": name,
            "n_seeds": m.get("n_seeds", 1),
            f"{KEY}_mean": mean,
            f"{KEY}_std": std,
            "ap_small_mean": _mean_std(m, "ap_small")[0],
            "latency_ms": _mean_std(m, "latency_ms_mean")[0],
            "fps": _mean_std(m, "fps_batch1")[0],
            "params_m": m.get("model", {}).get("params_m"),
            "gflops": m.get("model", {}).get("gflops"),
        })

    b_mean, b_std = _mean_std(base, KEY)
    for name, m in exps[1:]:
        mean, std = _mean_std(m, KEY)
        delta = round(mean - b_mean, 5)
        stds = [s for s in (b_std, std) if s is not None]
        if not stds:
            verdict = "VARIANCE-UNKNOWN (run 3 seeds before trusting this delta)"
        elif abs(delta) < max(stds):
            verdict = f"INCONCLUSIVE (|Δ|={abs(delta):.5f} < max std {max(stds):.5f})"
        else:
            verdict = "significant " + ("improvement" if delta > 0 else "regression")
        verdicts.append({"baseline": base_name, "candidate": name,
                         f"delta_{KEY}": delta, "verdict": verdict})

    per_class_delta = {}
    if len(exps) == 2:
        pc_a, pc_b = base.get("per_class", {}), exps[1][1].get("per_class", {})
        per_class_delta = {
            cls: round(pc_b[cls]["ap50_95"] - pc_a[cls]["ap50_95"], 5)
            for cls in pc_a if cls in pc_b
        }

    out = {
        "protocol_hash": hashes.pop(),
        "metric": KEY,
        "experiments": rows,
        "verdicts": verdicts,
        "per_class_delta": per_class_delta,
    }
    paths.COMPARISONS.mkdir(parents=True, exist_ok=True)
    stem = "cmp_" + "__".join(n for n, _ in exps)
    (paths.COMPARISONS / f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n")

    md = [f"# Comparison: {' vs '.join(n for n, _ in exps)}", "",
          f"Protocol: `{out['protocol_hash']}` | metric: {KEY}", "",
          "| exp | seeds | mAP50-95 (±std) | AP_small | ms/img | FPS | params(M) | GFLOPs |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        std_s = f" ±{r[f'{KEY}_std']}" if r[f"{KEY}_std"] is not None else ""
        md.append(f"| {r['exp']} | {r['n_seeds']} | {r[f'{KEY}_mean']}{std_s} | "
                  f"{r['ap_small_mean']} | {r['latency_ms']} | {r['fps']} | "
                  f"{r['params_m']} | {r['gflops']} |")
    md += ["", "## Verdicts", ""]
    md += [f"- **{v['candidate']}** vs {v['baseline']}: Δ{KEY}={v[f'delta_{KEY}']} → {v['verdict']}"
           for v in verdicts]
    (paths.COMPARISONS / f"{stem}.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {paths.COMPARISONS.relative_to(paths.ROOT)}/{stem}.{{json,md}}")
