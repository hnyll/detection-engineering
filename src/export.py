"""Deployment: export (ONNX / TensorRT engine / Core ML), pt-vs-ONNX parity,
and backend latency benchmarks. Artifacts land in <exp>/exports/ (weight formats
gitignored; manifest.json, parity_report.json and bench.json are committed).

Exports are keyed by seed and validated against a digest of their source
checkpoint via exports/manifest.json — a stale or wrong-seed file is never
silently reused."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import paths
from .data_coco import list_images
from .expmeta import load_config, load_protocol

PARITY_IMAGES = 64
IOU_MATCH = 0.9


def _preload_cudnn() -> None:
    """onnxruntime dlopens cuDNN sublibraries (libcudnn_adv.so.9 etc.) that
    torch's loader does not preload; load them RTLD_GLOBAL from the venv's
    nvidia wheels so the CUDA provider can start (WSL2 has no system cuDNN)."""
    import ctypes

    import torch

    base = Path(torch.__file__).parent.parent / "nvidia"
    for pat in ("cudnn/lib/libcudnn*.so.9", "cublas/lib/libcublas*.so.*"):
        for so in sorted(base.glob(pat)):  # core libcudnn.so.9 sorts first
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def _resolve_seed(exp_dir: Path, seed: int | None) -> int:
    trained = sorted(
        int(d.name[1:]) for d in (exp_dir / "seeds").glob("s*")
        if (d / "weights" / "best.pt").exists()
    )
    if seed is not None:
        if seed not in trained:
            raise SystemExit(f"{exp_dir.name}: no weights for seed {seed} (have {trained})")
        return seed
    if not trained:
        raise SystemExit(f"{exp_dir.name}: no trained weights — train first")
    return trained[0]


def _weights(exp_dir: Path, seed: int) -> Path:
    return exp_dir / "seeds" / f"s{seed}" / "weights" / "best.pt"


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.resolve().read_bytes()).hexdigest()[:16]


def _exports_dir(exp_dir: Path) -> Path:
    d = exp_dir / "exports"
    d.mkdir(exist_ok=True)
    return d


def _manifest_path(exp_dir: Path) -> Path:
    return _exports_dir(exp_dir) / "manifest.json"


def _load_manifest(exp_dir: Path) -> dict:
    p = _manifest_path(exp_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _target_name(seed: int, fmt: str, int8: bool) -> str:
    if fmt == "onnx":
        return f"best_s{seed}.onnx"
    if fmt == "engine":
        return f"best_s{seed}_{'int8' if int8 else 'fp16'}.engine"
    if fmt == "coreml":
        return f"best_s{seed}.mlpackage"
    raise SystemExit(f"unknown format {fmt}")


def export_model(exp_ref: str, fmt: str = "onnx", half: bool = False,
                 int8: bool = False, seed: int | None = None) -> Path:
    import torch  # noqa: F401  (must precede onnxruntime for bundled cuDNN)
    from ultralytics import YOLO

    _preload_cudnn()
    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    seed = _resolve_seed(exp_dir, seed)
    w = _weights(exp_dir, seed)
    target = _exports_dir(exp_dir) / _target_name(seed, fmt, int8)

    kwargs: dict = {"format": fmt, "imgsz": proto["imgsz"], "device": proto["device"]}
    if fmt == "engine":
        kwargs["half"] = not int8
        if int8:
            kwargs.update(int8=True, data=str(paths.dataset_yaml(cfg["meta"]["dataset"])),
                          fraction=0.1)
    elif fmt == "coreml":
        kwargs.update(half=True, nms=True)
    elif half:
        kwargs["half"] = True

    src_sha = _sha16(w)
    opts = {k: v for k, v in kwargs.items() if k != "device"}
    manifest = _load_manifest(exp_dir)
    entry = manifest.get(target.name)
    if target.exists() and entry \
            and entry.get("source_sha256") == src_sha and entry.get("opts") == opts:
        print(f"{target.relative_to(paths.ROOT)} up to date (source {src_sha})")
        return target
    if target.exists():
        print(f"{target.name} is stale (checkpoint or options changed) — re-exporting")

    print(f"== exporting {w} (seed {seed}, sha {src_sha}) -> {fmt}"
          + (" int8" if int8 else " fp16" if kwargs.get("half") else ""))
    produced = Path(YOLO(str(w)).export(**kwargs))
    if produced.resolve() != target.resolve():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        shutil.move(str(produced), str(target))

    manifest[target.name] = {
        "seed": seed,
        "source": str(w.resolve()),
        "source_sha256": src_sha,
        "opts": opts,
    }
    _manifest_path(exp_dir).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"== exported {target.relative_to(paths.ROOT)}")
    return target


def _boxes(result) -> list[tuple[list[float], float, int]]:
    return [
        ([float(v) for v in xyxy], float(conf), int(cls))
        for xyxy, conf, cls in zip(result.boxes.xyxy.tolist(),
                                   result.boxes.conf.tolist(),
                                   result.boxes.cls.tolist())
    ]


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parity(exp_ref: str, seed: int | None = None) -> None:
    """Numerical drift between the .pt model and its ONNX export, both run through
    the identical Ultralytics pre/post-processing at protocol settings."""
    import torch  # noqa: F401
    from ultralytics import YOLO

    _preload_cudnn()
    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    seed = _resolve_seed(exp_dir, seed)
    pt = _weights(exp_dir, seed)
    onnx_path = export_model(exp_ref, fmt="onnx", seed=seed)  # rebuilds if stale

    imgs = [str(p) for p in
            list_images(cfg["meta"]["dataset"], proto["split"])[:PARITY_IMAGES]]
    pargs = dict(imgsz=proto["imgsz"], conf=0.25, iou=proto["iou"],
                 max_det=proto["max_det"], device=proto["device"], verbose=False)

    m_pt, m_ox = YOLO(str(pt)), YOLO(str(onnx_path))
    n_pt = n_ox = matched = 0
    conf_ad, coord_ad = [], []
    for img in imgs:
        b_pt = _boxes(m_pt.predict(img, **pargs)[0])
        b_ox = _boxes(m_ox.predict(img, **pargs)[0])
        n_pt += len(b_pt)
        n_ox += len(b_ox)
        used = set()
        for box, conf, cls in b_pt:
            best_j, best_iou = -1, IOU_MATCH
            for j, (obox, oconf, ocls) in enumerate(b_ox):
                if j in used or ocls != cls:
                    continue
                iou = _iou_xyxy(box, obox)
                if iou >= best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0:
                used.add(best_j)
                matched += 1
                obox, oconf, _ = b_ox[best_j]
                conf_ad.append(abs(conf - oconf))
                coord_ad.extend(abs(a - b) for a, b in zip(box, obox))

    report = {
        "seed": seed,
        "weights": str(pt), "weights_sha256": _sha16(pt),
        "onnx": str(onnx_path.relative_to(paths.ROOT)),
        "n_images": len(imgs), "boxes_pt": n_pt, "boxes_onnx": n_ox,
        "matched": matched,
        "unmatched_pt": n_pt - matched, "unmatched_onnx": n_ox - matched,
        "conf_mad": round(sum(conf_ad) / len(conf_ad), 6) if conf_ad else None,
        "conf_max_ad": round(max(conf_ad), 6) if conf_ad else None,
        "coord_mad_px": round(sum(coord_ad) / len(coord_ad), 4) if coord_ad else None,
        "match_iou": IOU_MATCH, "conf_floor": 0.25,
    }
    del m_pt, m_ox
    from .train import _free_gpu
    _free_gpu()

    out = _exports_dir(exp_dir) / "parity_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(paths.ROOT)} (seed {seed})")
    print(f"   boxes pt/onnx {n_pt}/{n_ox}, matched {matched}, "
          f"conf MAD {report['conf_mad']}, coord MAD {report['coord_mad_px']}px")


BACKEND_SPECS = {
    "pt": None,
    "onnx": ("onnx", False, False),
    "trt-fp16": ("engine", True, False),
    "trt-int8": ("engine", False, True),
}


def bench(exp_ref: str, backend: str, seed: int | None = None) -> None:
    import torch  # noqa: F401
    from ultralytics import YOLO

    from .evaluate import measure_speed

    _preload_cudnn()
    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    seed = _resolve_seed(exp_dir, seed)

    if backend == "pt":
        target = _weights(exp_dir, seed)
    else:
        fmt, half, int8 = BACKEND_SPECS[backend]
        target = export_model(exp_ref, fmt=fmt, half=half, int8=int8, seed=seed)

    print(f"== bench {backend}: {target.name} (seed {seed})")
    model = YOLO(str(target))
    speed = measure_speed(model, cfg["meta"]["dataset"], proto["split"], proto)
    del model
    from .train import _free_gpu
    _free_gpu()

    out = _exports_dir(exp_dir) / "bench.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    results[backend] = {**speed, "file": target.name, "seed": seed, "imgsz": proto["imgsz"]}
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"   {speed['latency_ms_mean']}ms/img ({speed['fps_batch1']} FPS) "
          f"-> {out.relative_to(paths.ROOT)}")
