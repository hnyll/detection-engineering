"""Torchvision ConvNeXt-Tiny + FPN + Faster R-CNN experiment support.

This is intentionally separate from the Ultralytics path.  The model consumes
native-aspect images, lets GeneralizedRCNNTransform resize the long side to the
configured size, and returns detections in native image coordinates.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import resource
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from . import paths
from .data_coco import _label_path, build_gt, class_names, list_images
from .expmeta import (append_command_sh, file_sha16, load_state, save_state)

FAMILY = "torchvision_fasterrcnn_convnext"
ARCH = "convnext_tiny_fpn_fasterrcnn"


def is_convnext_frcnn(cfg: dict) -> bool:
    return (cfg.get("meta") or {}).get("model_family") == FAMILY


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pretrained_weights_path() -> Path:
    filename = ConvNeXt_Tiny_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def validate_config(cfg: dict) -> dict:
    train = dict(cfg.get("train") or {})
    if train.get("architecture") != ARCH:
        raise SystemExit(f"{FAMILY} requires train.architecture: {ARCH}")
    if int(train.get("batch", 1)) != 1:
        raise SystemExit("ConvNeXt/Faster R-CNN is initially restricted to physical batch 1 on this 8 GB GPU")
    anchors = tuple(int(v) for v in train.get("anchor_sizes", [8, 16, 32, 64, 128]))
    if len(anchors) != 5 or any(v <= 0 for v in anchors):
        raise SystemExit("train.anchor_sizes must contain five positive P2-P6 sizes")
    train["anchor_sizes"] = anchors
    train["imgsz"] = int(train.get("imgsz", 960))
    train["epochs"] = int(train.get("epochs", 24))
    train["batch"] = 1
    train["accumulate"] = int(train.get("accumulate", 4))
    train["workers"] = min(int(train.get("workers", 2)), 2)
    train["lr"] = float(train.get("lr", 1e-4))
    train["weight_decay"] = float(train.get("weight_decay", 0.05))
    train["horizontal_flip"] = float(train.get("horizontal_flip", 0.5))
    train["amp"] = bool(train.get("amp", True))
    train["score_thresh"] = float(train.get("score_thresh", 0.001))
    train["nms_iou"] = float(train.get("nms_iou", 0.7))
    train["max_det"] = int(train.get("max_det", 300))
    if train["accumulate"] <= 0 or train["epochs"] <= 0 or train["imgsz"] <= 0:
        raise SystemExit("epochs, imgsz, and accumulate must be positive")
    if not 0 <= train["horizontal_flip"] <= 1:
        raise SystemExit("horizontal_flip must be in [0, 1]")
    return train


class YoloBoxDataset(Dataset):
    """Read native images and YOLO labels for torchvision detection models."""

    def __init__(self, images: list[Path], *, augment: bool, flip_p: float = 0.0):
        self.images = [Path(p) for p in images]
        self.augment = augment
        self.flip_p = flip_p

    def __len__(self) -> int:
        return len(self.images)

    def label_count(self, index: int) -> int:
        p = _label_path(self.images[index])
        return len(p.read_text().splitlines()) if p.exists() else 0

    def __getitem__(self, index: int):
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes, labels = [], []
        label_path = _label_path(path)
        if label_path.exists():
            for row in label_path.read_text().splitlines():
                fields = row.split()
                if len(fields) < 5:
                    continue
                cls = int(fields[0])
                cx, cy, bw, bh = (float(v) for v in fields[1:5])
                x1 = max(0.0, (cx - bw / 2) * width)
                y1 = max(0.0, (cy - bh / 2) * height)
                x2 = min(float(width), (cx + bw / 2) * width)
                y2 = min(float(height), (cy + bh / 2) * height)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # zero is Faster R-CNN background

        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        image_t = TF.pil_to_tensor(image).float().div_(255.0)
        if self.augment and torch.rand(()) < self.flip_p:
            image_t = torch.flip(image_t, dims=(2,))
            if boxes_t.numel():
                old_x1 = boxes_t[:, 0].clone()
                old_x2 = boxes_t[:, 2].clone()
                boxes_t[:, 0] = width - old_x2
                boxes_t[:, 2] = width - old_x1

        area = ((boxes_t[:, 2] - boxes_t[:, 0]) *
                (boxes_t[:, 3] - boxes_t[:, 1])) if boxes_t.numel() else torch.zeros(0)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
        }
        return image_t, target


def _collate(batch):
    return tuple(zip(*batch))


def _manifest_images(dataset_name: str) -> list[Path]:
    cfg_path = paths.dataset_yaml(dataset_name)
    data = yaml.safe_load(cfg_path.read_text()) or {}
    ref = data.get("train")
    if not isinstance(ref, str):
        raise SystemExit(f"{cfg_path} train must be a file-backed manifest")
    root = Path(data.get("path") or cfg_path.parent)
    if not root.is_absolute():
        root = (cfg_path.parent / root).resolve()
    manifest = Path(ref)
    if not manifest.is_absolute():
        manifest = root / manifest
    images = [Path(row.strip()) for row in manifest.read_text().splitlines() if row.strip()]
    if not images or any(not p.is_file() for p in images):
        raise SystemExit(f"invalid or empty training manifest: {manifest}")
    return images


def build_model(cfg: dict, *, pretrained: bool) -> FasterRCNN:
    train = validate_config(cfg)
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    base = convnext_tiny(weights=weights)
    backbone = BackboneWithFPN(
        base.features,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )
    anchor_sizes = tuple((value,) for value in train["anchor_sizes"])
    ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchors = AnchorGenerator(anchor_sizes, ratios)
    roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone,
        num_classes=7,
        min_size=train["imgsz"],
        max_size=train["imgsz"],
        rpn_anchor_generator=anchors,
        box_roi_pool=roi_pool,
        box_score_thresh=train["score_thresh"],
        box_nms_thresh=train["nms_iou"],
        box_detections_per_img=train["max_det"],
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=300,
        rpn_nms_thresh=train["nms_iou"],
    )


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _device_targets(targets, device):
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets]


def _loader(dataset: Dataset, train: dict, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=train["batch"],
        shuffle=shuffle,
        num_workers=train["workers"],
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=train["workers"] > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def hardware_smoke(exp_ref: str, batches: int = 10, seed: int = 17) -> None:
    """Real-data forward/loss/backward/optimizer smoke without experiment weights."""
    from .expmeta import load_config

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    if not is_convnext_frcnn(cfg):
        raise SystemExit(f"{exp_dir.name} is not a {FAMILY} experiment")
    train = validate_config(cfg)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this smoke from the host GPU context")
    if batches <= 0:
        raise SystemExit("batches must be positive")
    append_command_sh(
        exp_dir,
        ["model-smoke", exp_dir.name, "--batches", str(batches), "--seed", str(seed)],
    )

    weights_path = pretrained_weights_path()
    expected = (cfg.get("meta") or {}).get("initial_weights_sha256")
    if not weights_path.exists() or (expected and _sha256(weights_path) != expected):
        raise SystemExit("cached ConvNeXt pretrained weights are absent or do not match the experiment pin")

    paths_all = _manifest_images(cfg["meta"]["train_dataset"])
    # Dense images first; deduplicate so the report shows a useful range rather
    # than repeated copies of the same repeat-factor sample.
    unique = list(dict.fromkeys(paths_all))
    unique.sort(key=lambda p: len(_label_path(p).read_text().splitlines()), reverse=True)
    selected = unique[: max(batches * train["batch"], train["batch"])]
    dataset = YoloBoxDataset(selected, augment=True, flip_p=train["horizontal_flip"])
    loader = _loader(dataset, train, shuffle=False, seed=seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    model = build_model(cfg, pretrained=True).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"],
                                  weight_decay=train["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=train["amp"])
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_reserved = torch.cuda.memory_reserved()

    step_times, losses, object_counts = [], [], []
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader):
        if step >= batches:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        target_gpu = _device_targets(targets, device)
        object_counts.append(sum(len(target["labels"]) for target in targets))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=train["amp"]):
            loss_dict = model(images, target_gpu)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss at step {step}: {loss_dict}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)
        losses.append(float(loss.detach().cpu()))
        print(
            f"smoke {step + 1}/{batches}: objects={object_counts[-1]} "
            f"loss={losses[-1]:.4f} step={step_times[-1]:.1f}ms "
            f"peak_reserved={torch.cuda.max_memory_reserved() / 2**20:.0f}MiB"
        )

    elapsed = time.perf_counter() - started
    import psutil
    report = {
        "schema_version": 1,
        "status": "pass",
        "experiment": exp_dir.name,
        "model_family": FAMILY,
        "architecture": ARCH,
        "seed": seed,
        "batches": len(step_times),
        "imgsz": train["imgsz"],
        "batch": train["batch"],
        "amp": train["amp"],
        "weights": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "parameters": model_parameter_count(model),
        "object_counts": object_counts,
        "losses": losses,
        "step_times_ms": step_times,
        "steady_step_ms_median": statistics.median(step_times[1:] or step_times),
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "baseline_reserved_mib": baseline_reserved / 2**20,
        "process_rss_mib": psutil.Process().memory_info().rss / 2**20,
        "gpu": torch.cuda.get_device_name(0),
        "notes": [
            "Real images and labels, standard torchvision transform, horizontal flip only.",
            "Dense unique manifest images are intentionally evaluated first.",
            "This smoke does not create or modify experiment checkpoints.",
        ],
    }
    out = exp_dir / "artifacts" / f"hardware_smoke_s{seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


@torch.inference_mode()
def predict_dataset(model, dataset_name: str, split: str, device, *, amp: bool = True):
    dataset = YoloBoxDataset(list_images(dataset_name, split), augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=_collate, pin_memory=True)
    detections = []
    model.eval()
    progress = tqdm(
        loader, total=len(loader), desc=f"{split} inference",
        unit="image", dynamic_ncols=True, leave=False,
    )
    for images, targets in progress:
        images_gpu = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            outputs = model(images_gpu)
        image_id = int(targets[0]["image_id"])
        output = outputs[0]
        for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box.detach().cpu())
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(score), 5),
            })
    return detections


def _checkpoint_payload(model, optimizer, scheduler, scaler, *, epoch, best_map, seed, cfg):
    return {
        "schema_version": 1,
        "model_family": FAMILY,
        "architecture": ARCH,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_map50_95": best_map,
        "seed": seed,
        "config_sha256": file_sha16(
            paths.resolve_exp(cfg["meta"]["name"]) / "config.yaml"
        ),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint_model(cfg: dict, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_family") != FAMILY or payload.get("architecture") != ARCH:
        raise RuntimeError(f"not a {ARCH} checkpoint: {checkpoint}")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state"])
    return model.to(device), payload


def train_experiment(exp_ref: str, seeds: list[int] | None = None, resume: bool = False,
                     interrupt_after: int | None = None, epochs: int | None = None) -> None:
    """Full per-seed custom detector trainer with epoch-level resume."""
    from .evaluate import coco_metrics
    from .expmeta import load_config

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    train = validate_config(cfg)
    if epochs is not None:
        train["epochs"] = int(epochs)
    seeds = seeds or cfg["meta"]["seeds"]
    train_images = _manifest_images(cfg["meta"]["train_dataset"])
    val_dataset_name = cfg["meta"]["dataset"]
    gt_json = build_gt(val_dataset_name, "val")
    names = class_names(val_dataset_name)
    weights_path = pretrained_weights_path()
    expected = cfg["meta"].get("initial_weights_sha256")
    if not weights_path.exists() or (expected and _sha256(weights_path) != expected):
        raise SystemExit("ConvNeXt pretrained checkpoint is missing or differs from the config pin")

    argv = ["train", exp_dir.name]
    if seeds != cfg["meta"]["seeds"]:
        argv += ["--seeds", ",".join(map(str, seeds))]
    if resume:
        argv.append("--resume")
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    append_command_sh(exp_dir, argv)

    for seed in seeds:
        from .train import _attest_training_inputs, _setup_wandb
        _attest_training_inputs(
            cfg,
            paths.dataset_yaml(cfg["meta"]["train_dataset"]),
            str(weights_path),
            exp_dir,
            seed,
        )
        wandb_mode = _setup_wandb()
        wandb_run = None
        if wandb_mode != "disabled":
            import wandb
            wandb_run = wandb.init(
                project="detection-engineering",
                name=f"{paths.run_name(exp_dir, seed)}{'_resume' if resume else ''}",
                dir=str(paths.ROOT),
                config={"model_family": FAMILY, **train},
            )
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise SystemExit("ConvNeXt/Faster R-CNN full training requires CUDA")
        model = build_model(cfg, pretrained=not resume).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"],
                                      weight_decay=train["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train["epochs"], eta_min=train["lr"] * 0.01
        )
        scaler = torch.amp.GradScaler("cuda", enabled=train["amp"])
        rdir = paths.run_dir(exp_dir, seed)
        last_path = rdir / "weights" / "last.pt"
        best_path = rdir / "weights" / "best.pt"
        start_epoch, best_map = 0, -1.0
        if resume:
            if not last_path.exists():
                raise SystemExit(f"--resume: no checkpoint at {last_path}")
            payload = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            scaler.load_state_dict(payload["scaler_state"])
            start_epoch = int(payload["epoch"]) + 1
            best_map = float(payload.get("best_map50_95", -1.0))
            random.setstate(payload["rng"]["python"])
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
            print(f"== {exp_dir.name} s{seed}: resume epoch {start_epoch + 1}")

        dataset = YoloBoxDataset(train_images, augment=True, flip_p=train["horizontal_flip"])
        loader = _loader(dataset, train, shuffle=True, seed=seed + start_epoch)
        csv_path = rdir / "results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch", "train_loss", "map50_95", "map50", "ap_small",
                      "lr", "epoch_seconds", "peak_reserved_mib"]
        write_header = not csv_path.exists() or not resume
        csv_file = csv_path.open("a" if resume else "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        interrupted = False
        try:
            for epoch in range(start_epoch, train["epochs"]):
                model.train()
                torch.cuda.reset_peak_memory_stats()
                epoch_start = time.perf_counter()
                running = []
                optimizer.zero_grad(set_to_none=True)
                progress = tqdm(
                    loader,
                    total=len(loader),
                    desc=f"epoch {epoch + 1}/{train['epochs']}",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=True,
                )
                for step, (images, targets) in enumerate(progress):
                    images = [image.to(device, non_blocking=True) for image in images]
                    target_gpu = _device_targets(targets, device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=train["amp"]):
                        loss_dict = model(images, target_gpu)
                        loss = sum(loss_dict.values())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss e{epoch + 1} step{step}: {loss_dict}")
                    scaler.scale(loss / train["accumulate"]).backward()
                    boundary = ((step + 1) % train["accumulate"] == 0 or
                                step + 1 == len(loader))
                    if boundary:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    running.append(float(loss.detach().cpu()))
                    if step == 0 or (step + 1) % 10 == 0:
                        progress.set_postfix(
                            loss=f"{statistics.fmean(running[-50:]):.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                            vram=f"{torch.cuda.max_memory_reserved() / 2**20:.0f}MiB",
                        )
                scheduler.step()
                detections = predict_dataset(model, val_dataset_name, "val", device,
                                              amp=train["amp"])
                acc = coco_metrics(gt_json, detections, train["max_det"], names)
                score = acc["overall"]["map50_95"]
                payload = _checkpoint_payload(
                    model, optimizer, scheduler, scaler, epoch=epoch,
                    best_map=max(best_map, score), seed=seed, cfg=cfg
                )
                _save_checkpoint(last_path, payload)
                if score > best_map:
                    best_map = score
                    _save_checkpoint(best_path, payload)
                row = {
                    "epoch": epoch + 1,
                    "train_loss": statistics.fmean(running),
                    "map50_95": score,
                    "map50": acc["overall"]["map50"],
                    "ap_small": acc["overall"]["ap_small"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                }
                writer.writerow(row)
                csv_file.flush()
                if wandb_run is not None:
                    wandb_run.log(row, step=epoch + 1)
                print(
                    f"e{epoch + 1}: mAP50-95={score:.5f} APs={acc['overall']['ap_small']:.5f} "
                    f"best={best_map:.5f} time={row['epoch_seconds'] / 60:.1f}m"
                )
                if interrupt_after is not None and epoch + 1 >= interrupt_after:
                    interrupted = True
                    break
        except KeyboardInterrupt:
            interrupted = True
            print(f"cancelled; resume from completed epoch checkpoint {last_path}")
        finally:
            csv_file.close()
            if wandb_run is not None:
                wandb_run.finish()

        state = load_state(exp_dir)
        state["runs"].append({
            "seed": seed,
            "run_dir": str(rdir.relative_to(paths.ROOT)),
            "status": "interrupted" if interrupted else "completed",
            "model_family": FAMILY,
            "wandb_mode": wandb_mode,
            "wandb_id": wandb_run.id if wandb_run is not None else None,
            "checkpoint": str(last_path),
            "best_map50_95": best_map,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "timestamp": time.time(),
        })
        save_state(exp_dir, state)
        if not interrupted:
            seed_weights = paths.seed_dir(exp_dir, seed) / "weights"
            seed_weights.mkdir(exist_ok=True)
            for name in ("best.pt", "last.pt"):
                src = rdir / "weights" / name
                dst = seed_weights / name
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())
        del model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()
        if interrupted:
            return


def predict_to_coco_checkpoint(cfg: dict, checkpoint: Path, dataset_name: str,
                               split: str, out_json: Path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_model(cfg, checkpoint, device)
    train = validate_config(cfg)
    detections = predict_dataset(model, dataset_name, split, device, amp=train["amp"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(detections))
    return model, detections


@torch.inference_mode()
def measure_speed(model, dataset_name: str, split: str, proto: dict, amp: bool = True):
    device = next(model.parameters()).device
    images = list_images(dataset_name, split)
    lat = proto.get("latency") or {}
    warmup, timed = int(lat.get("warmup", 20)), int(lat.get("timed", 100))
    sequence = (images * math.ceil((warmup + timed) / len(images)))[:warmup + timed]
    times = []
    model.eval()
    for index, path in enumerate(sequence):
        with Image.open(path) as source:
            image = TF.pil_to_tensor(source.convert("RGB")).float().div_(255.0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            model([image])
        if device.type == "cuda":
            torch.cuda.synchronize()
        if index >= warmup:
            times.append((time.perf_counter() - started) * 1000)
    times.sort()
    mean = statistics.fmean(times)
    return {
        "latency_ms_mean": round(mean, 2),
        "latency_ms_p50": round(statistics.median(times), 2),
        "latency_ms_p95": round(times[max(0, int(len(times) * 0.95) - 1)], 2),
        "fps_batch1": round(1000 / mean, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_timed": len(times),
    }
