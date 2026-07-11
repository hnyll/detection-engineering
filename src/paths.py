"""Repo path resolution and experiment id allocation."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TEMPLATES = ROOT / "templates"
EXPERIMENTS = ROOT / "experiments"
COMPARISONS = EXPERIMENTS / "comparisons"
INTERVIEW = ROOT / "interview"
RUNS = ROOT / "runs"
DATA = ROOT / "data"
COCO_GT_DIR = DATA / "coco_gt"

PROTOCOL_YAML = CONFIGS / "protocol.yaml"

_EXP_RE = re.compile(r"^exp_(\d{3})_(.+)$")


def dataset_yaml(dataset: str) -> Path:
    p = CONFIGS / f"{dataset}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no dataset config: {p}")
    return p


def list_experiments() -> list[Path]:
    if not EXPERIMENTS.exists():
        return []
    return sorted(d for d in EXPERIMENTS.iterdir() if d.is_dir() and _EXP_RE.match(d.name))


def next_exp_id() -> int:
    ids = [int(_EXP_RE.match(d.name).group(1)) for d in list_experiments()]
    return max(ids) + 1 if ids else 0


def resolve_exp(ref: str) -> Path:
    """Accept 'exp_001_baseline', '001', '1', or a unique name fragment."""
    exps = list_experiments()
    if (EXPERIMENTS / ref).is_dir():
        return EXPERIMENTS / ref
    if ref.isdigit():
        wanted = int(ref)
        for d in exps:
            if int(_EXP_RE.match(d.name).group(1)) == wanted:
                return d
    matches = [d for d in exps if ref in d.name]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"cannot resolve experiment '{ref}' "
        f"({'ambiguous: ' + ', '.join(m.name for m in matches) if matches else 'no match'})"
    )


def seed_dir(exp_dir: Path, seed: int) -> Path:
    d = exp_dir / "seeds" / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_name(exp_dir: Path, seed: int) -> str:
    return f"{exp_dir.name}_s{seed}"


def run_dir(exp_dir: Path, seed: int) -> Path:
    return RUNS / run_name(exp_dir, seed)
