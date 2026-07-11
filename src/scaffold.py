"""`new NAME` — create experiments/exp_NNN_NAME from templates/."""

from __future__ import annotations

import re

from .paths import EXPERIMENTS, TEMPLATES, next_exp_id


def new_experiment(name: str) -> None:
    name = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name):
        raise SystemExit(f"experiment name must be [a-z0-9_], got '{name}'")

    exp = f"exp_{next_exp_id():03d}_{name}"
    exp_dir = EXPERIMENTS / exp
    exp_dir.mkdir(parents=True)
    (exp_dir / "artifacts").mkdir()
    (exp_dir / "seeds").mkdir()

    for tpl in ("hypothesis.md", "failure_report.md", "decision.md", "config.yaml"):
        text = (TEMPLATES / tpl).read_text()
        text = text.replace("{EXP}", exp).replace("{NAME}", name)
        (exp_dir / tpl).write_text(text)

    print(f"created {exp_dir.relative_to(EXPERIMENTS.parent)}")
    print("next: write the Prediction in hypothesis.md BEFORE training, then edit config.yaml")
