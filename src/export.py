"""Deployment: export (ONNX / TensorRT engine / Core ML), pt-vs-ONNX parity,
and backend latency benchmarks. All artifacts land in <exp>/exports/ (weights
formats are gitignored; parity_report.json and bench.json are committed)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import paths
from .data_coco import list_images
from .expmeta import load_config, load_protocol

PARITY_IMAGES = 64
IOU_MATCH = 0.9


def _weights(exp_dir: Path, seed: int | None) -> Path:
    if seed is None:
        cands = sorted((exp_dir / "seeds").glob("s*/weights/best.pt"))
        if not cands:
            raise SystemExit(f"{exp_dir.name}: no trained weights — train first")
        return cands[0]
    w = exp_dir / "seeds" / f"s{seed}" / "weights" / "best.pt"
    if not w.exists():
        raise SystemExit(f"no weights for seed {seed}")
    return w


def _exports_dir(exp_dir: Path) -> Path:
    d = exp_dir / "exports"
    d.mkdir(exist_ok=True)
    return d


def _target_name(fmt: str, half: bool, int8: bool) -> str:
    if fmt == "onnx":
        return "best.onnx"
    if fmt == "engine":
        return "best_int8.engine" if int8 else "best_fp16.engine"
    if fmt == "coreml":
        return "best.mlpackage"
    raise SystemExit(f"unknown format {fmt}")


def export_model(exp_ref: str, fmt: str = "onnx", half: bool = False,
                 int8: bool = False, seed: int | None = None) -> Path:
    import torch  # noqa: F401  (must precede onnxruntime for bundled cuDNN)
    from ultralytics import YOLO

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    w = _weights(exp_dir, seed)
    target = _exports_dir(exp_dir) / _target_name(fmt, half, int8)
    if target.exists():
        print(f"{target.relative_to(paths.ROOT)} already exists")
        return target

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

    print(f"== exporting {w} -> {fmt}" + (" int8" if int8 else " fp16" if kwargs.get("half") else ""))
    produced = Path(YOLO(str(w)).export(**kwargs))
    if produced.resolve() != target.resolve():
        if target.exists() or (target.is_dir() and fmt == "coreml"):
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(produced), str(target))
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

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    pt = _weights(exp_dir, seed)
    onnx_path = _exports_dir(exp_dir) / "best.onnx"
    if not onnx_path.exists():
        export_model(exp_ref, fmt="onnx", seed=seed)

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
        "weights": str(pt), "onnx": str(onnx_path.relative_to(paths.ROOT)),
        "n_images": len(imgs), "boxes_pt": n_pt, "boxes_onnx": n_ox,
        "matched": matched,
        "unmatched_pt": n_pt - matched, "unmatched_onnx": n_ox - matched,
        "conf_mad": round(sum(conf_ad) / len(conf_ad), 6) if conf_ad else None,
        "conf_max_ad": round(max(conf_ad), 6) if conf_ad else None,
        "coord_mad_px": round(sum(coord_ad) / len(coord_ad), 4) if coord_ad else None,
        "match_iou": IOU_MATCH, "conf_floor": 0.25,
    }
    out = _exports_dir(exp_dir) / "parity_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(paths.ROOT)}")
    print(f"   boxes pt/onnx {n_pt}/{n_ox}, matched {matched}, "
          f"conf MAD {report['conf_mad']}, coord MAD {report['coord_mad_px']}px")


BACKEND_FILES = {
    "pt": ("best.pt", None),
    "onnx": ("best.onnx", ("onnx", False, False)),
    "trt-fp16": ("best_fp16.engine", ("engine", True, False)),
    "trt-int8": ("best_int8.engine", ("engine", False, True)),
}


def bench(exp_ref: str, backend: str, seed: int | None = None) -> None:
    import torch  # noqa: F401
    from ultralytics import YOLO

    from .evaluate import measure_speed

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    proto = load_protocol()
    fname, export_spec = BACKEND_FILES[backend]

    if backend == "pt":
        target = _weights(exp_dir, seed)
    else:
        target = _exports_dir(exp_dir) / fname
        if not target.exists():
            fmt, half, int8 = export_spec
            export_model(exp_ref, fmt=fmt, half=half, int8=int8, seed=seed)

    print(f"== bench {backend}: {target.name}")
    speed = measure_speed(YOLO(str(target)), cfg["meta"]["dataset"], proto["split"], proto)

    out = _exports_dir(exp_dir) / "bench.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    results[backend] = {**speed, "file": fname, "imgsz": proto["imgsz"]}
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"   {speed['latency_ms_mean']}ms/img ({speed['fps_batch1']} FPS) "
          f"-> {out.relative_to(paths.ROOT)}")
