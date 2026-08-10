"""Ultralytics train wrapper: machine-safe defaults, seed control, resume,
W&B online/offline fallback, and evidence recording for the Train gate."""

from __future__ import annotations

import hashlib
import json
import os
import resource
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import paths
from .expmeta import append_command_sh, file_sha16, load_config, load_state, save_state

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


class _PlannedInterrupt(Exception):
    """Raised by the phase0 resume-proof callback. Distinct from KeyboardInterrupt
    so a user's Ctrl-C (even during a phase0 run) is never mistaken for the plan."""


def _wandb_mode() -> str:
    """online if credentials exist, else offline (never blocks a run)."""
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        return "online"
    return "offline"


def _setup_wandb() -> str:
    # Some restricted environments cannot create W&B's local service socket.
    # Allow a fully disabled mode for local training; this also avoids exporting
    # metadata when the caller explicitly sets WANDB_MODE=disabled.
    if os.environ.get("WANDB_MODE", "").lower() == "disabled" or os.environ.get("DE_DISABLE_WANDB"):
        os.environ["WANDB_MODE"] = "disabled"
        from ultralytics import settings
        settings.update({"wandb": False})
        return "disabled"
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


def _free_gpu() -> None:
    """Collect + empty CUDA cache. Callers MUST `del` their model reference
    first — deleting a parameter here would only drop the local binding, leaving
    the caller's reference (and the whole trainer graph behind it) alive into
    the next seed's model construction. Back-to-back live models OOM the 8GB card."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_rss_mb() -> dict:
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"self": round(self_kb / 1024), "max_child": round(child_kb / 1024)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_evidence(path: Path, payload: dict) -> None:
    """Write immutable experiment evidence, accepting an identical retry."""
    if path.exists():
        try:
            if json.loads(path.read_text()) == payload:
                return
        except json.JSONDecodeError:
            pass
        raise RuntimeError(
            f"refusing to overwrite different evidence at {path}; "
            "use a new experiment or remove the stale artifact deliberately"
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _training_manifest(data_yaml: Path) -> Path:
    """Resolve a file-backed Ultralytics training manifest from its data YAML."""
    data = yaml.safe_load(data_yaml.read_text()) or {}
    train_ref = data.get("train")
    if not isinstance(train_ref, str):
        raise SystemExit(
            f"pinned train_manifest_sha256 requires a single file-backed train entry in {data_yaml}"
        )
    root = Path(data.get("path") or data_yaml.parent).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    manifest = Path(train_ref).expanduser()
    if not manifest.is_absolute():
        manifest = root / manifest
    if not manifest.is_file():
        raise SystemExit(f"pinned training manifest does not exist: {manifest}")
    return manifest.resolve()


def _attest_training_inputs(
    cfg: dict, data_yaml: Path, model_ref: str, exp_dir: Path, seed: int
) -> None:
    """Fail before training if an experiment's declared input hashes drift."""
    meta = cfg.get("meta") or {}
    expected_model = meta.get("initial_weights_sha256")
    expected_manifest = meta.get("train_manifest_sha256")
    if not expected_model and not expected_manifest:
        return

    evidence: dict = {"seed": seed, "checks": {}}
    mismatches = []
    if expected_model:
        model_path = Path(model_ref).expanduser()
        if not model_path.is_absolute():
            model_path = paths.ROOT / model_path
        if not model_path.is_file():
            raise SystemExit(f"pinned initial weights do not exist: {model_path}")
        actual = _sha256(model_path.resolve())
        evidence["checks"]["initial_weights"] = {
            "path": str(model_path.resolve()),
            "expected_sha256": expected_model,
            "actual_sha256": actual,
            "match": actual == expected_model,
        }
        if actual != expected_model:
            mismatches.append("initial weights")

    if expected_manifest:
        manifest = _training_manifest(data_yaml)
        actual = _sha256(manifest)
        evidence["checks"]["train_manifest"] = {
            "path": str(manifest),
            "expected_sha256": expected_manifest,
            "actual_sha256": actual,
            "match": actual == expected_manifest,
        }
        if actual != expected_manifest:
            mismatches.append("training manifest")

    artifact = exp_dir / "artifacts" / f"training_inputs_s{seed}.json"
    artifact.parent.mkdir(exist_ok=True)
    _write_json_evidence(artifact, evidence)
    if mismatches:
        raise SystemExit(f"training input hash mismatch: {', '.join(mismatches)}")


def _persist_class_weight_evidence(
    trainer,
    exp_dir: Path,
    seed: int,
    data_yaml: Path,
    expected: dict | None = None,
) -> None:
    """Record the class counts and weights actually installed by Ultralytics."""
    power = float(getattr(trainer.args, "cls_pw", 0.0))
    if power == 0.0:
        if expected:
            raise RuntimeError(
                "preregistered class weights exist but effective trainer cls_pw is 0"
            )
        return

    import numpy as np
    import ultralytics

    labels = trainer.train_loader.dataset.labels
    classes = np.concatenate([label["cls"].reshape(-1) for label in labels]).astype(int)
    nc = int(trainer.data["nc"])
    counts = np.bincount(classes, minlength=nc).astype(int)
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    weights_tensor = getattr(model, "class_weights", None)
    if weights_tensor is None:
        raise RuntimeError("cls_pw is enabled but Ultralytics did not install class_weights")
    weights = [float(value) for value in weights_tensor.detach().cpu().tolist()]

    raw_names = trainer.data["names"]
    if isinstance(raw_names, dict):
        names = [str(raw_names.get(i, raw_names.get(str(i), i))) for i in range(nc)]
    else:
        names = [str(name) for name in raw_names]
    actual = dict(zip(names, weights))
    expected = expected or {}
    missing = sorted(set(expected) - set(actual))
    zero_count_classes = [names[i] for i, value in enumerate(counts) if value == 0]
    max_abs_diff = max(
        (abs(actual[name] - float(value)) for name, value in expected.items() if name in actual),
        default=0.0,
    )
    verified = not missing and not zero_count_classes and max_abs_diff <= 0.002

    manifest = _training_manifest(data_yaml)

    evidence = {
        "seed": seed,
        "ultralytics_version": ultralytics.__version__,
        "cls_pw": power,
        "formula": "normalize_mean((1 / class_count) ** cls_pw)",
        "data_yaml": str(data_yaml.resolve()),
        "train_manifest": str(manifest),
        "train_manifest_sha256": _sha256(manifest),
        "training_entries": len(labels),
        "class_counts": dict(zip(names, [int(value) for value in counts.tolist()])),
        "class_weights": actual,
        "expected_class_weights": expected,
        "zero_count_classes": zero_count_classes,
        "max_abs_diff": max_abs_diff,
        "verified": verified,
    }
    artifact = exp_dir / "artifacts" / f"class_weights_s{seed}.json"
    artifact.parent.mkdir(exist_ok=True)
    _write_json_evidence(artifact, evidence)
    if not verified:
        details = (
            f"missing classes={missing}, zero-count classes={zero_count_classes}, "
            f"max_abs_diff={max_abs_diff:.6f}"
        )
        raise RuntimeError(f"runtime class weights do not match the preregistered values: {details}")


def train(exp_ref: str, seeds: list[int] | None = None, resume: bool = False,
          interrupt_after: int | None = None, epochs: int | None = None) -> None:
    """Train each requested seed. `interrupt_after=N` raises KeyboardInterrupt after
    epoch N (used by phase0 to prove checkpoint resume works)."""
    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    if cfg["meta"].get("analysis_only"):
        raise SystemExit(
            f"{exp_dir.name} is analysis-only — run its command.sh workflow; "
            "do not train it"
        )
    if cfg["meta"].get("inference_only"):
        raise SystemExit(
            f"{exp_dir.name} is inference-only — evaluate its inherited checkpoint; "
            "do not train it"
        )
    from .convnext_frcnn import is_convnext_frcnn
    if is_convnext_frcnn(cfg):
        from .convnext_frcnn import train_experiment
        return train_experiment(
            exp_ref, seeds=seeds, resume=resume,
            interrupt_after=interrupt_after, epochs=epochs,
        )
    from ultralytics import YOLO

    seeds = seeds or cfg["meta"]["seeds"]
    # Evaluation always uses meta.dataset. An experiment may point training at a
    # derived view (for example, a repeat-factor sampling manifest) without
    # changing the official validation dataset or its comparison identity.
    train_dataset = cfg["meta"].get("train_dataset", cfg["meta"]["dataset"])
    data_yaml = paths.dataset_yaml(train_dataset)
    wandb_mode = _setup_wandb()

    argv = ["train", exp_dir.name]
    if seeds != cfg["meta"]["seeds"]:
        argv += ["--seeds", ",".join(map(str, seeds))]
    if resume:
        argv += ["--resume"]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    if interrupt_after is not None:
        argv += ["--interrupt-after", str(interrupt_after)]
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

        resume_sha = None
        if resume:
            if epochs is not None:
                raise SystemExit("--epochs cannot change on --resume: ultralytics "
                                 "restores the checkpoint's args, so the override "
                                 "would be silently ignored — start a new run instead")
            last = rdir / "weights" / "last.pt"
            if not last.exists():
                raise SystemExit(f"--resume: no checkpoint at {last}")
            resume_sha = file_sha16(last)  # digest of the INPUT checkpoint, pre-overwrite
            model = YOLO(str(last))
            print(f"== {exp_dir.name} seed {seed}: resuming from {last} (ckpt {resume_sha})")
        else:
            model_ref = str(train_args.pop("model", "yolo11n.pt"))
            _attest_training_inputs(cfg, data_yaml, model_ref, exp_dir, seed)
            model = YOLO(model_ref)
            print(f"== {exp_dir.name} seed {seed}: training ({wandb_mode} W&B)")

        if interrupt_after is not None:
            def _interrupt(trainer, _n=interrupt_after):
                if trainer.epoch + 1 >= _n:  # trainer.epoch is 0-indexed
                    raise _PlannedInterrupt(f"planned interrupt after epoch {_n}")
            model.add_callback("on_fit_epoch_end", _interrupt)

        expected_class_weights = (cfg.get("meta") or {}).get("expected_class_weights")
        configured_cls_pw = float((cfg.get("train") or {}).get("cls_pw", 0.0))
        if expected_class_weights or configured_cls_pw > 0.0:
            def _record_class_weights(
                trainer,
                _exp_dir=exp_dir,
                _seed=seed,
                _data_yaml=data_yaml,
                _expected=expected_class_weights,
            ):
                _persist_class_weight_evidence(
                    trainer, _exp_dir, _seed, _data_yaml, _expected
                )
            model.add_callback("on_pretrain_routine_end", _record_class_weights)

        # Own the W&B run: the ultralytics callback derives the project name from
        # args.project (our absolute runs path -> garbage name). It skips its own
        # init when a run is already active and logs into ours instead.
        wb_run = None
        run_id = None
        if wandb_mode != "disabled":
            import wandb
            wb_run = wandb.init(
                project="detection-engineering",
                name=paths.run_name(exp_dir, seed) + ("_resume" if resume else ""),
                dir=str(paths.ROOT),
            )
            run_id = wb_run.id

        before = _list_wandb_runs()
        interrupted = planned = False
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
        except _PlannedInterrupt:
            interrupted = planned = True
            print(f"== planned interrupt (checkpoint at {rdir / 'weights' / 'last.pt'})")
        except KeyboardInterrupt:
            interrupted = True
            print(f"== cancelled by user (checkpoint at {rdir / 'weights' / 'last.pt'})")
        finally:
            if wb_run is not None:  # normal completion is finished by the callback
                wb_run.finish()

        new_wandb = sorted(_list_wandb_runs() - before)
        state = load_state(exp_dir)
        state["runs"].append({
            "seed": seed,
            "run_dir": str(rdir.relative_to(paths.ROOT)),
            "status": "interrupted" if interrupted else "completed",
            "planned_interrupt": planned if interrupted else None,
            "resumed_from": ({"path": str(rdir / "weights" / "last.pt"),
                              "sha256": resume_sha} if resume else None),
            "wandb_mode": wandb_mode,
            "wandb_id": run_id,
            "wandb_runs": new_wandb,
            "peak_rss_mb": _peak_rss_mb(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        save_state(exp_dir, state)

        if interrupted and not planned:
            # genuine Ctrl-C: state is recorded, now propagate the cancellation
            # instead of silently moving on to the next seed
            raise KeyboardInterrupt

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

        del model
        _free_gpu()
