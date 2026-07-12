"""Compare experiments under the identical evaluation protocol.

Refuses to compare across protocol hashes. Verdict logic per the spec's
statistical discipline: an improvement smaller than the observed seed variance
is INCONCLUSIVE; with fewer than 2 seeds on both sides the variance is unknown
and the verdict says so."""

from __future__ import annotations

import json

from . import paths
from .expmeta import file_sha16, protocol_hash

KEY = "map50_95"
MIN_SEEDS = 3  # spec: variance measured with 3 seeds for important comparisons


def _load(exp_ref: str) -> tuple[str, dict]:
    exp_dir = paths.resolve_exp(exp_ref)
    mj = exp_dir / "metrics.json"
    if not mj.exists():
        raise SystemExit(f"{exp_dir.name}: no metrics.json — run eval first")
    m = json.loads(mj.read_text())
    # metrics must describe the checkpoints that exist NOW — a retrained seed
    # invalidates its old numbers
    for sk, sv in (m.get("seeds") or {}).items():
        recorded = sv.get("weights_sha256")
        w = exp_dir / "seeds" / sk / "weights" / "best.pt"
        if recorded and w.exists() and file_sha16(w) != recorded:
            raise SystemExit(
                f"REFUSED: {exp_dir.name} {sk} was retrained after its eval "
                f"(checkpoint digest changed) — re-run eval first"
            )
    return exp_dir.name, m


def _mean_std(m: dict, key: str) -> tuple[float | None, float | None]:
    agg = m.get("aggregate", {})
    return agg.get(f"{key}_mean"), agg.get(f"{key}_std")


def compare(exp_refs: list[str]) -> None:
    exps = [_load(r) for r in exp_refs]
    if len(exps) < 2:
        raise SystemExit("need at least two experiments to compare")

    # comparability identity = protocol hash + dataset + split (+ GT fingerprint
    # when recorded) — a hash match alone would happily compare COCO128 vs VisDrone
    idents = {(m["protocol"]["hash"], m["protocol"]["dataset"], m["protocol"]["split"],
               m["protocol"].get("gt")) for _, m in exps}
    if len(idents) > 1:
        for name, m in exps:
            p = m["protocol"]
            print(f"  {name}: protocol {p['hash']} dataset {p['dataset']}/{p['split']} "
                  f"gt {p.get('gt')}")
        raise SystemExit(
            "REFUSED: evaluation identities differ (protocol/dataset/split/GT) — "
            "these results are not comparable."
        )
    if protocol_hash() != next(iter(idents))[0]:
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
    n_base = base.get("n_seeds", 1)
    for name, m in exps[1:]:
        mean, std = _mean_std(m, KEY)
        n_cand = m.get("n_seeds", 1)
        delta = round(mean - b_mean, 5)
        if b_std is None or std is None or n_base < MIN_SEEDS or n_cand < MIN_SEEDS:
            verdict = (f"INSUFFICIENT-SEEDS (significance needs ≥{MIN_SEEDS} seeds per "
                       f"side; have {n_base} vs {n_cand})")
        elif delta == 0:
            verdict = "NO DIFFERENCE"
        elif abs(delta) <= max(b_std, std):
            verdict = f"INCONCLUSIVE (|Δ|={abs(delta):.5f} ≤ max std {max(b_std, std):.5f})"
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

    ident = next(iter(idents))
    out = {
        "protocol_hash": ident[0],
        "dataset": ident[1],
        "split": ident[2],
        "metric": KEY,
        "experiments": rows,
        "verdicts": verdicts,
        "per_class_delta": per_class_delta,
    }
    paths.COMPARISONS.mkdir(parents=True, exist_ok=True)
    stem = "cmp_" + "__".join(n for n, _ in exps)
    (paths.COMPARISONS / f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n")

    md = [f"# Comparison: {' vs '.join(n for n, _ in exps)}", "",
          f"Protocol: `{out['protocol_hash']}` | dataset: {out['dataset']}/{out['split']} "
          f"| metric: {KEY}", "",
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
