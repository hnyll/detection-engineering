#!/usr/bin/env python3
"""Isolated RT-DETR hardware feasibility benchmark.

This script never creates or mutates an experiment. It supports two checks:

1. batch-1 end-to-end PyTorch inference over evenly spaced VisDrone images;
2. one synthetic AMP training step, including gradients, AdamW state, and EMA.

The training step can vary the number of GT objects because RT-DETR's
denoising queries make dense VisDrone images materially more memory-intensive.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/rtdetr-hardware-bench/yolo")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rtdetr-hardware-bench/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "data/derived/visdrone_exp006_coco_aligned/images/val"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_result(result: dict, output: Path | None) -> None:
    text = json.dumps(result, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output}")
    print(text, end="")


def _cuda_identity(torch) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_vram_mib": round(props.total_memory / 2**20, 2),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _base_result(args, torch, ultralytics, weights: Path) -> dict:
    return {
        "schema_version": 2,
        "mode": args.mode,
        "model": args.model,
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "weights_bytes": weights.stat().st_size,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "gpu": _cuda_identity(torch),
        "notes": [
            "Hardware-feasibility measurement only; pretrained COCO predictions are not VisDrone accuracy results.",
            "PyTorch execution; no TensorRT, export, or architecture-specific deployment optimization.",
        ],
    }


def _load_wrapper(model_path: str):
    from ultralytics import RTDETR, YOLO

    return RTDETR(model_path) if "rtdetr" in Path(model_path).name.lower() else YOLO(model_path)


def _evenly_spaced_images(root: Path, count: int) -> list[Path]:
    images = sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"no images under {root}")
    count = min(count, len(images))
    if count == 1:
        return [images[0]]
    indices = [round(i * (len(images) - 1) / (count - 1)) for i in range(count)]
    return [images[i] for i in indices]


def benchmark_inference(args, torch, ultralytics, weights: Path) -> dict:
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    wrapper = _load_wrapper(str(weights))
    params = get_num_params(wrapper.model)
    gflops = get_flops(wrapper.model, imgsz=args.imgsz)
    images = _evenly_spaced_images(Path(args.images), args.samples)
    predict_args = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": 0,
        "verbose": False,
        "save": False,
        "half": False,
    }

    # Predictor construction, model transfer, and kernel selection are excluded.
    for i in range(args.warmup):
        wrapper.predict(str(images[i % len(images)]), **predict_args)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    times_ms: list[float] = []
    detections: list[int] = []
    for image in images:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = wrapper.predict(str(image), **predict_args)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)
        detections.append(len(result[0].boxes))

    ordered = sorted(times_ms)
    mean = statistics.fmean(times_ms)
    result = _base_result(args, torch, ultralytics, weights)
    result.update(
        {
            "status": "pass",
            "params": params,
            "params_m": round(params / 1e6, 3),
            "gflops": round(float(gflops), 2),
            "precision": "fp32",
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "warmup": args.warmup,
            "timed_images": len(images),
            "image_sampling": "evenly spaced over sorted six-class VisDrone validation view",
            "latency_ms": {
                "mean": round(mean, 3),
                "p50": round(statistics.median(ordered), 3),
                "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
                "fps_from_mean": round(1000.0 / mean, 2),
            },
            "detections": {
                "mean": round(statistics.fmean(detections), 2),
                "min": min(detections),
                "max": max(detections),
            },
            "cuda_memory_mib": {
                "baseline_allocated": round(baseline_allocated / 2**20, 2),
                "baseline_reserved": round(baseline_reserved / 2**20, 2),
                "peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                "peak_reserved": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            },
        }
    )
    return result


def _custom_rtdetr_model(weights: Path, nc: int):
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel

    pretrained = RTDETR(str(weights))
    model = RTDETRDetectionModel(deepcopy(pretrained.model.yaml), nc=nc, verbose=False)
    model.load(pretrained.model)
    # DetectionTrainer normally attaches these dataset attributes before the
    # first loss call. The isolated step creates the model directly.
    model.nc = nc
    model.names = {i: f"class_{i}" for i in range(nc)}
    del pretrained
    return model


def _synthetic_batch(torch, batch_size: int, imgsz: int, objects: int, nc: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(17)
    img = torch.rand((batch_size, 3, imgsz, imgsz), generator=generator, dtype=torch.float32).cuda()
    batch_idx = torch.arange(batch_size).repeat_interleave(objects)
    cls = torch.randint(0, nc, (batch_size * objects, 1), generator=generator)
    centers = torch.rand((batch_size * objects, 2), generator=generator) * 0.9 + 0.05
    sizes = torch.rand((batch_size * objects, 2), generator=generator) * 0.15 + 0.01
    return {
        "img": img,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": torch.cat((centers, sizes), dim=1),
    }


def benchmark_train_step(args, torch, ultralytics, weights: Path) -> dict:
    from ultralytics.utils.torch_utils import ModelEMA, get_num_params

    result = _base_result(args, torch, ultralytics, weights)
    result.update(
        {
            "objects_per_image": args.objects,
            "steps": args.steps,
            "nc": args.nc,
            "precision": "amp-fp16",
            "includes": ["forward", "RT-DETR loss", "backward", "AdamW first step", "EMA update"],
            "limitations": [
                "Synthetic tensors avoid dataloader variability but exercise the real six-class RT-DETR loss.",
                "Does not prove a full epoch or every possible CUDA kernel will succeed.",
                "Dense-label memory is explicitly tested because RT-DETR creates denoising queries from max GT count.",
            ],
        }
    )
    try:
        model = _custom_rtdetr_model(weights, args.nc).cuda().train()
        params = get_num_params(model)
        ema = ModelEMA(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        batch = _synthetic_batch(torch, args.batch, args.imgsz, args.objects, args.nc)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        step_times_ms: list[float] = []
        total_loss = loss_items = None
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss, loss_items = model(batch)
                total_loss = loss.sum()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            torch.cuda.synchronize()
            step_times_ms.append((time.perf_counter() - start) * 1000)

        import psutil

        steady = step_times_ms[1:] or step_times_ms

        result.update(
            {
                "status": "pass",
                "params": params,
                "params_m": round(params / 1e6, 3),
                "step_times_ms": [round(x, 3) for x in step_times_ms],
                "first_step_ms": round(step_times_ms[0], 3),
                "steady_step_ms_mean": round(statistics.fmean(steady), 3),
                "loss": round(float(total_loss.detach().cpu()), 6),
                "loss_items": [round(float(x), 6) for x in loss_items.detach().cpu()],
                "host_rss_mib": round(psutil.Process().memory_info().rss / 2**20, 2),
                "cuda_memory_mib": {
                    "baseline_allocated": round(baseline_allocated / 2**20, 2),
                    "baseline_reserved": round(baseline_reserved / 2**20, 2),
                    "peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                    "peak_reserved": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                },
            }
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        result.update(
            {
                "status": "oom",
                "error": str(exc),
                "cuda_memory_mib": {
                    "peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20, 2),
                    "peak_reserved": round(torch.cuda.max_memory_reserved() / 2**20, 2),
                },
            }
        )
    finally:
        for name in ("batch", "optimizer", "ema", "model"):
            if name in locals():
                del locals()[name]
        gc.collect()
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "train-step"), required=True)
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--objects", type=int, default=100)
    parser.add_argument("--nc", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.batch <= 0 or args.samples <= 0 or args.warmup < 0:
        parser.error("imgsz, batch, and samples must be positive; warmup must be nonnegative")
    if args.objects <= 0 or args.nc <= 0 or args.steps <= 0:
        parser.error("objects, nc, and steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark in the host GPU context")
    torch.backends.cudnn.benchmark = True
    weights = Path(args.model)
    if not weights.exists():
        # Ultralytics downloads named official weights during wrapper creation.
        wrapper = _load_wrapper(args.model)
        del wrapper
        weights = Path(args.model)
    if not weights.exists():
        raise FileNotFoundError(f"model weights were not created: {weights}")
    result = (
        benchmark_inference(args, torch, ultralytics, weights)
        if args.mode == "inference"
        else benchmark_train_step(args, torch, ultralytics, weights)
    )
    _write_result(result, args.output)


if __name__ == "__main__":
    main()
