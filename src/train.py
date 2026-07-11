"""Ultralytics train wrapper: machine-safe defaults, seed control, resume,
W&B online/offline fallback, and evidence recording for the Train gate."""

from __future__ import annotations

import os
import resource
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .expmeta import append_command_sh, load_config, load_state, save_state

# Hard machine-safety defaults for the 8GB 4060 / 11GB WSL2 box (docs/machine_notes.md).
# Experiment config.yaml `train:` values are applied on top, but workers/cache are
# clamped — exceeding them has frozen this machine before.
SAFE_DEFAULTS = {
    "workers": 2,
    "amp": True,
    "cache": False,
    "batch": 16,
    "device": 0,
    "deterministic": True,
}
MAX_WORKERS = 2


def _wandb_mode() -> str:
    """online if credentials exist, else offline (never blocks a run)."""
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        return "online"
    return "offline"


def _setup_wandb() -> str:
    mode = _wandb_mode()
    if mode == "offline":
        os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "detection-engineering")
    from ultralytics import settings
    settings.update({"wandb": True})
    return mode


def _list_wandb_runs() -> set:
    wdir = paths.ROOT / "wandb"
    return {p.name for p in wdir.glob("*run-*")} if wdir.exists() else set()


def _peak_rss_mb() -> dict:
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"self": round(self_kb / 1024), "max_child": round(child_kb / 1024)}


def train(exp_ref: str, seeds: list[int] | None = None, resume: bool = False,
          interrupt_after: int | None = None, epochs: int | None = None) -> None:
    """Train each requested seed. `interrupt_after=N` raises KeyboardInterrupt after
    epoch N (used by phase0 to prove checkpoint resume works)."""
    from ultralytics import YOLO

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    seeds = seeds or cfg["meta"]["seeds"]
    data_yaml = paths.dataset_yaml(cfg["meta"]["dataset"])
    wandb_mode = _setup_wandb()

    argv = ["train", exp_dir.name]
    if seeds != cfg["meta"]["seeds"]:
        argv += ["--seeds", ",".join(map(str, seeds))]
    if resume:
        argv += ["--resume"]
    append_command_sh(exp_dir, argv)

    for seed in seeds:
        rdir = paths.run_dir(exp_dir, seed)
        train_args = dict(SAFE_DEFAULTS)
        train_args.update(cfg.get("train", {}))
        if train_args.get("workers", 0) > MAX_WORKERS:
            print(f"clamping workers {train_args['workers']} -> {MAX_WORKERS} (machine safety)")
            train_args["workers"] = MAX_WORKERS
        train_args["cache"] = False
        if epochs is not None:
            train_args["epochs"] = epochs

        if resume:
            last = rdir / "weights" / "last.pt"
            if not last.exists():
                raise SystemExit(f"--resume: no checkpoint at {last}")
            model = YOLO(str(last))
            print(f"== {exp_dir.name} seed {seed}: resuming from {last}")
        else:
            model = YOLO(train_args.pop("model", "yolo11n.pt"))
            print(f"== {exp_dir.name} seed {seed}: training ({wandb_mode} W&B)")

        if interrupt_after is not None:
            def _interrupt(trainer, _n=interrupt_after):
                if trainer.epoch + 1 >= _n:  # trainer.epoch is 0-indexed
                    raise KeyboardInterrupt(f"harness: planned interrupt after epoch {_n}")
            model.add_callback("on_fit_epoch_end", _interrupt)

        before = _list_wandb_runs()
        interrupted = False
        try:
            if resume:
                model.train(resume=True)
            else:
                model.train(
                    data=str(data_yaml),
                    seed=seed,
                    project=str(paths.RUNS),
                    name=paths.run_name(exp_dir, seed),
                    exist_ok=True,
                    **{k: v for k, v in train_args.items() if k != "model"},
                )
        except KeyboardInterrupt:
            interrupted = True
            print(f"== interrupted (checkpoint at {rdir / 'weights' / 'last.pt'})")

        new_wandb = sorted(_list_wandb_runs() - before)
        state = load_state(exp_dir)
        state["runs"].append({
            "seed": seed,
            "run_dir": str(rdir.relative_to(paths.ROOT)),
            "status": "interrupted" if interrupted else "completed",
            "resumed_from": str(rdir / "weights" / "last.pt") if resume else None,
            "wandb_mode": wandb_mode,
            "wandb_runs": new_wandb,
            "peak_rss_mb": _peak_rss_mb(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        save_state(exp_dir, state)

        if not interrupted:
            sdir = paths.seed_dir(exp_dir, seed) / "weights"
            sdir.mkdir(exist_ok=True)
            for w in ("best.pt", "last.pt"):
                src, dst = rdir / "weights" / w, sdir / w
                if src.exists():
                    dst.unlink(missing_ok=True)
                    dst.symlink_to(src.resolve())
            rss = state["runs"][-1]["peak_rss_mb"]
            print(f"== done seed {seed}; peak RSS self={rss['self']}MB child={rss['max_child']}MB")
            if wandb_mode == "offline" and new_wandb:
                print(f"   offline W&B run(s): {new_wandb} — sync later with: uv run wandb sync wandb/<run>")
