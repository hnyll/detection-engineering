"""Experiment metadata: config.yaml, protocol hashing, decision.md frontmatter,
command.sh reproduction records, and per-experiment state.json."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .paths import PROTOCOL_YAML, ROOT


# -- config.yaml --------------------------------------------------------------

def load_config(exp_dir: Path) -> dict:
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    if not isinstance(cfg, dict) or "meta" not in cfg or "train" not in cfg:
        raise SystemExit(f"{exp_dir.name}/config.yaml must contain 'meta' and 'train' sections")
    cfg["meta"].setdefault("seeds", [17])
    cfg["meta"].setdefault("dataset", "visdrone")
    return cfg


# -- fixed evaluation protocol -------------------------------------------------

def load_protocol() -> dict:
    return yaml.safe_load(PROTOCOL_YAML.read_text())


def protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL_YAML.read_bytes()).hexdigest()[:16]


def file_sha16(p: Path | str) -> str:
    """Content digest for artifact provenance (checkpoints, prediction files)."""
    return hashlib.sha256(Path(p).resolve().read_bytes()).hexdigest()[:16]


def stale_seeds(exp_dir: Path, metrics: dict) -> list[str]:
    """Seeds whose recorded checkpoint digest no longer matches the checkpoint on
    disk — their metrics describe a model that no longer exists."""
    out = []
    for sk, sv in (metrics.get("seeds") or {}).items():
        recorded = sv.get("weights_sha256")
        w = exp_dir / "seeds" / sk / "weights" / "best.pt"
        if recorded and w.exists() and file_sha16(w) != recorded:
            out.append(sk)
    return out


# -- decision.md frontmatter ---------------------------------------------------

def read_frontmatter(md_path: Path) -> dict:
    if not md_path.exists():
        return {}
    text = md_path.read_text()
    if not text.startswith("---"):
        return {}
    try:
        block = text.split("---", 2)[1]
        return yaml.safe_load(block) or {}
    except Exception:
        return {}


def write_frontmatter(md_path: Path, updates: dict) -> None:
    text = md_path.read_text()
    if not text.startswith("---"):
        raise SystemExit(f"{md_path} has no frontmatter block")
    _, block, body = text.split("---", 2)
    fm = yaml.safe_load(block) or {}
    fm.update(updates)
    md_path.write_text("---\n" + yaml.safe_dump(fm, sort_keys=False) + "---" + body)


# -- reproduction record -------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def env_versions() -> dict:
    import torch
    import ultralytics
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "git_sha": git_sha(),
    }


def append_command_sh(exp_dir: Path, argv: list[str]) -> None:
    """Record the exact reproduction command with context. Appends — the file is
    the experiment's full command history."""
    path = exp_dir / "command.sh"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = env_versions()
    lines = []
    if not path.exists():
        lines.append("#!/usr/bin/env bash\n# Reproduction commands (appended by the harness)\n")
    lines.append(
        f"\n# {stamp} | git {versions['git_sha']} | torch {versions['torch']} "
        f"| ultralytics {versions['ultralytics']}\n"
        + shlex.join(["uv", "run", "python", "-m", "src.cli", *argv]) + "\n"
    )
    with path.open("a") as f:
        f.writelines(lines)


# -- state.json ----------------------------------------------------------------

def load_state(exp_dir: Path) -> dict:
    p = exp_dir / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"runs": []}


def save_state(exp_dir: Path, state: dict) -> None:
    (exp_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
