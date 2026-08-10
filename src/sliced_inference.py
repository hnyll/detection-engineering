"""Deterministic source-image tiling for an inference-only experiment.

The detector still receives ``imgsz`` from the fixed protocol. Native source
images are cut into overlapping windows, each window is predicted separately,
boxes are shifted back to source coordinates, and a class-aware global NMS
keeps the protocol's final ``max_det`` candidates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path


PIPELINE_SCHEMA_VERSION = 2


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _implementation_sha16() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def normalized_tiling(cfg: dict, proto: dict) -> dict | None:
    """Validate and fully materialize an experiment's tiling configuration."""
    raw = ((cfg.get("evaluation") or {}).get("tiling") or {})
    if not raw or not raw.get("enabled", False):
        return None
    if not isinstance(raw, dict):
        raise SystemExit("evaluation.tiling must be a mapping")

    allowed = {
        "enabled", "tile_height", "tile_width", "overlap_height_ratio",
        "overlap_width_ratio", "include_full_image", "tile_batch",
        "per_tile_conf", "per_tile_nms_iou", "per_tile_max_det", "merge",
        "merge_iou", "global_max_det",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise SystemExit(f"unknown evaluation.tiling key(s): {sorted(unknown)}")

    out = {
        "enabled": True,
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "implementation": "sahi.get_slice_bboxes+torchvision.batched_nms",
        "implementation_sha256": _implementation_sha16(),
        "dependencies": {
            "sahi": _distribution_version("sahi"),
            "torchvision": _distribution_version("torchvision"),
            "opencv": _distribution_version("opencv-python", "opencv-python-headless"),
        },
        "tile_height": int(raw.get("tile_height", 640)),
        "tile_width": int(raw.get("tile_width", 640)),
        "overlap_height_ratio": float(raw.get("overlap_height_ratio", 0.2)),
        "overlap_width_ratio": float(raw.get("overlap_width_ratio", 0.2)),
        "include_full_image": bool(raw.get("include_full_image", False)),
        "tile_batch": int(raw.get("tile_batch", 1)),
        "per_tile_conf": float(raw.get("per_tile_conf", proto["conf"])),
        "per_tile_nms_iou": float(raw.get("per_tile_nms_iou", proto["iou"])),
        "per_tile_max_det": int(raw.get("per_tile_max_det", proto["max_det"])),
        "merge": str(raw.get("merge", "class_aware_nms")),
        "merge_iou": float(raw.get("merge_iou", proto["iou"])),
        "global_max_det": int(raw.get("global_max_det", proto["max_det"])),
    }
    if out["tile_height"] <= 0 or out["tile_width"] <= 0:
        raise SystemExit("tile dimensions must be positive")
    for key in ("overlap_height_ratio", "overlap_width_ratio"):
        if not 0 <= out[key] < 1:
            raise SystemExit(f"{key} must be in [0, 1)")
    if out["tile_batch"] != 1:
        raise SystemExit("the current tiled evaluator supports tile_batch=1 only")
    if out["merge"] != "class_aware_nms":
        raise SystemExit("the current tiled evaluator supports merge=class_aware_nms only")

    # These equality checks prevent a tile experiment from silently bundling a
    # confidence, NMS, or detection-budget change into the intended treatment.
    fixed = {
        "per_tile_conf": float(proto["conf"]),
        "per_tile_nms_iou": float(proto["iou"]),
        "per_tile_max_det": int(proto["max_det"]),
        "global_max_det": int(proto["max_det"]),
    }
    for key, expected in fixed.items():
        if out[key] != expected:
            raise SystemExit(
                f"evaluation.tiling.{key}={out[key]} must equal protocol value {expected}; "
                "test that setting in a separate experiment"
            )
    if out["merge_iou"] != float(proto["iou"]):
        raise SystemExit("evaluation.tiling.merge_iou must equal the fixed protocol NMS IoU")
    if not 0 <= out["per_tile_conf"] <= 1:
        raise SystemExit("evaluation.tiling.per_tile_conf must be in [0, 1]")
    for key in ("per_tile_nms_iou", "merge_iou"):
        if not 0 <= out[key] <= 1:
            raise SystemExit(f"evaluation.tiling.{key} must be in [0, 1]")
    if out["per_tile_max_det"] <= 0 or out["global_max_det"] <= 0:
        raise SystemExit("evaluation.tiling detection caps must be positive")
    return out


def _shift_clip_box(
    xyxy: list[float], x_offset: int, y_offset: int, width: int, height: int
) -> list[float] | None:
    """Map one tile-local xyxy box to a clipped source-image box."""
    import math

    lx1, ly1, lx2, ly2 = (float(v) for v in xyxy)
    gx1 = min(float(width), max(0.0, lx1 + x_offset))
    gy1 = min(float(height), max(0.0, ly1 + y_offset))
    gx2 = min(float(width), max(0.0, lx2 + x_offset))
    gy2 = min(float(height), max(0.0, ly2 + y_offset))
    if not all(math.isfinite(v) for v in (gx1, gy1, gx2, gy2)):
        return None
    if gx2 <= gx1 or gy2 <= gy1:
        return None
    return [gx1, gy1, gx2, gy2]


def _slice_boxes(height: int, width: int, tiling: dict) -> list[list[int]]:
    """Return the canonical SAHI tile geometry used by both pipeline variants."""
    from sahi.slicing import get_slice_bboxes

    return get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=tiling["tile_height"],
        slice_width=tiling["tile_width"],
        auto_slice_resolution=False,
        overlap_height_ratio=tiling["overlap_height_ratio"],
        overlap_width_ratio=tiling["overlap_width_ratio"],
    )


def _merge_candidates(
    boxes: list[list[float]], scores: list[float], classes: list[int],
    merge_iou: float, max_det: int, sources: list[str] | None = None,
) -> tuple[list[dict], int]:
    """Class-aware global NMS with a deterministic score-ranked final cap."""
    import torch
    from torchvision.ops import batched_nms

    if not boxes:
        return [], 0
    if sources is not None and len(sources) != len(boxes):
        raise ValueError("sources must align one-to-one with candidate boxes")
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    classes_t = torch.tensor(classes, dtype=torch.int64)
    keep = batched_nms(boxes_t, scores_t, classes_t, merge_iou)
    post_merge = int(keep.numel())
    keep = keep[:max_det]
    predictions = []
    for i in keep.tolist():
        row = {
            "xyxy": boxes_t[i].tolist(),
            "score": float(scores_t[i]),
            "cls": int(classes_t[i]),
        }
        if sources is not None:
            row["source"] = sources[i]
        predictions.append(row)
    return predictions, post_merge


def _predict_image_tiled(model, image_path: Path, proto: dict, tiling: dict) -> tuple[list[dict], dict]:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    tile_boxes = _slice_boxes(height, width, tiling)

    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_classes: list[int] = []
    all_sources: list[str] = []
    border_candidates = 0
    invalid_boxes = 0
    local_rank_cap_scores: list[float] = []
    for x1, y1, x2, y2 in tile_boxes:
        tile = np.ascontiguousarray(image[y1:y2, x1:x2])
        result = model.predict(
            tile,
            batch=1,
            rect=False,
            imgsz=proto["imgsz"],
            conf=tiling["per_tile_conf"],
            iou=tiling["per_tile_nms_iou"],
            max_det=tiling["per_tile_max_det"],
            device=proto["device"],
            stream=False,
            verbose=False,
        )[0]
        tile_h, tile_w = tile.shape[:2]
        if len(result.boxes) >= tiling["per_tile_max_det"]:
            local_rank_cap_scores.append(float(result.boxes.conf.min().item()))
        for xyxy, score, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            lx1, ly1, lx2, ly2 = (float(v) for v in xyxy)
            if lx1 <= 1 or ly1 <= 1 or lx2 >= tile_w - 1 or ly2 >= tile_h - 1:
                border_candidates += 1
            mapped = _shift_clip_box([lx1, ly1, lx2, ly2], x1, y1, width, height)
            if mapped is None or not np.isfinite(score):
                invalid_boxes += 1
                continue
            all_boxes.append(mapped)
            all_scores.append(float(score))
            all_classes.append(int(cls))
            all_sources.append("tile")

    full_frame_candidates = 0
    full_frame_rank_cap_score = None
    if tiling["include_full_image"]:
        result = model.predict(
            np.ascontiguousarray(image),
            batch=1,
            rect=False,
            imgsz=proto["imgsz"],
            conf=tiling["per_tile_conf"],
            iou=tiling["per_tile_nms_iou"],
            max_det=tiling["per_tile_max_det"],
            device=proto["device"],
            stream=False,
            verbose=False,
        )[0]
        if len(result.boxes) >= tiling["per_tile_max_det"]:
            full_frame_rank_cap_score = float(result.boxes.conf.min().item())
        for xyxy, score, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            mapped = _shift_clip_box(
                [float(v) for v in xyxy], 0, 0, width, height
            )
            if mapped is None or not np.isfinite(score):
                invalid_boxes += 1
                continue
            all_boxes.append(mapped)
            all_scores.append(float(score))
            all_classes.append(int(cls))
            all_sources.append("full_frame")
            full_frame_candidates += 1

    if not all_boxes:
        return [], {
            "width": width, "height": height, "tiles": tile_boxes,
            "pre_merge": 0, "post_merge": 0, "final": 0,
            "border_candidates": border_candidates,
            "rejected_invalid_candidates": invalid_boxes,
            "invalid_final_boxes": 0,
            "local_rank_cap_scores": local_rank_cap_scores,
            "full_frame_candidates": full_frame_candidates,
            "full_frame_rank_cap_score": full_frame_rank_cap_score,
            "model_forwards": len(tile_boxes) + int(tiling["include_full_image"]),
            "final_tile_detections": 0,
            "final_full_frame_detections": 0,
        }

    predictions, post_merge = _merge_candidates(
        all_boxes, all_scores, all_classes,
        tiling["merge_iou"], tiling["global_max_det"], all_sources,
    )
    final_tile = sum(row.get("source") == "tile" for row in predictions)
    final_full_frame = sum(row.get("source") == "full_frame" for row in predictions)
    return predictions, {
        "width": width,
        "height": height,
        "tiles": tile_boxes,
        "pre_merge": len(all_boxes),
        "post_merge": post_merge,
        "final": len(predictions),
        "border_candidates": border_candidates,
        "rejected_invalid_candidates": invalid_boxes,
        "invalid_final_boxes": 0,
        "local_rank_cap_scores": local_rank_cap_scores,
        "full_frame_candidates": full_frame_candidates,
        "full_frame_rank_cap_score": full_frame_rank_cap_score,
        "model_forwards": len(tile_boxes) + int(tiling["include_full_image"]),
        "final_tile_detections": final_tile,
        "final_full_frame_detections": final_full_frame,
    }


def full_frame_parity(model, image_path: Path, proto: dict, tiling: dict) -> dict:
    """Prove file-path and ndarray view inference have identical preprocessing.

    Global merge is measured separately: it is an explicit tiled-pipeline
    treatment and can legitimately suppress an additional already-local-NMSed
    candidate, so it is not part of the parity pass/fail condition.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    direct = model.predict(
        str(image_path), batch=1, rect=False, imgsz=proto["imgsz"],
        conf=proto["conf"], iou=proto["iou"], max_det=proto["max_det"],
        device=proto["device"], stream=False, verbose=False,
    )[0]
    array_view = model.predict(
        np.ascontiguousarray(image), batch=1, rect=False, imgsz=proto["imgsz"],
        conf=tiling["per_tile_conf"], iou=tiling["per_tile_nms_iou"],
        max_det=tiling["per_tile_max_det"], device=proto["device"],
        stream=False, verbose=False,
    )[0]
    direct_rows = list(zip(
        direct.boxes.xyxy.tolist(), direct.boxes.conf.tolist(), direct.boxes.cls.tolist()
    ))
    array_rows = list(zip(
        array_view.boxes.xyxy.tolist(),
        array_view.boxes.conf.tolist(),
        array_view.boxes.cls.tolist(),
    ))
    count_match = len(direct_rows) == len(array_rows)
    max_box_delta = max_score_delta = 0.0
    class_match = count_match
    if count_match:
        for (box, score, cls), (array_box, array_score, array_cls) in zip(direct_rows, array_rows):
            max_box_delta = max(
                max_box_delta,
                max(abs(float(a) - float(b)) for a, b in zip(box, array_box)),
            )
            max_score_delta = max(max_score_delta, abs(float(score) - float(array_score)))
            class_match = class_match and int(cls) == int(array_cls)
    merged, _ = _merge_candidates(
        [row[0] for row in array_rows],
        [float(row[1]) for row in array_rows],
        [int(row[2]) for row in array_rows],
        tiling["merge_iou"], tiling["global_max_det"],
    )
    passed = count_match and class_match and max_box_delta <= 1e-5 and max_score_delta <= 1e-7
    report = {
        "passed": passed,
        "image": image_path.name,
        "native_shape": [height, width],
        "direct_predictions": len(direct_rows),
        "one_window_predictions": len(array_rows),
        "one_window_count": 1,
        "post_global_merge_predictions": len(merged),
        "global_merge_removed": len(array_rows) - len(merged),
        "class_order_match": class_match,
        "max_box_delta": max_box_delta,
        "max_score_delta": max_score_delta,
        "tolerances": {"box": 1e-5, "score": 1e-7},
    }
    if not passed:
        raise SystemExit(
            "tiled evaluator view-preprocessing parity failed — refusing accuracy evaluation: "
            + json.dumps(report, sort_keys=True)
        )
    return report


def full_frame_parity_suite(model, image_paths: list[Path], proto: dict, tiling: dict) -> dict:
    """Run one-window parity on one representative image per native shape."""
    import cv2

    selected: dict[tuple[int, int], Path] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        shape = tuple(int(v) for v in image.shape[:2])
        selected.setdefault(shape, image_path)
    cases = [full_frame_parity(model, path, proto, tiling) for path in selected.values()]
    return {
        "passed": all(case["passed"] for case in cases),
        "selection": "first sorted validation image for each native (height, width)",
        "cases": cases,
    }


def predict_to_coco_tiled(
    model,
    dataset: str,
    split: str,
    proto: dict,
    tiling: dict,
    out_json: Path,
    manifest_json: Path,
    diagnostics_json: Path,
) -> tuple[list[dict], dict]:
    """Run pure tiled inference over a split and emit globally merged COCO boxes."""
    from .data_coco import list_images

    images = list_images(dataset, split)
    detections: list[dict] = []
    manifest = []
    totals = defaultdict(int)
    tile_counts = []
    model_forward_counts = []
    local_rank_cap_scores: list[float] = []
    full_frame_rank_cap_scores: list[float] = []
    images_with_full_frame_survivor = 0
    for image_id, image_path in enumerate(images):
        predictions, diag = _predict_image_tiled(model, image_path, proto, tiling)
        tile_counts.append(len(diag["tiles"]))
        model_forward_counts.append(int(diag["model_forwards"]))
        manifest.append({
            "image_id": image_id,
            "file_name": image_path.name,
            "width": diag["width"],
            "height": diag["height"],
            "tiles": diag["tiles"],
        })
        for key in (
            "pre_merge", "post_merge", "final", "border_candidates",
            "rejected_invalid_candidates", "invalid_final_boxes",
            "full_frame_candidates", "final_tile_detections",
            "final_full_frame_detections",
        ):
            totals[key] += int(diag[key])
        if diag["final_full_frame_detections"]:
            images_with_full_frame_survivor += 1
        local_rank_cap_scores.extend(diag["local_rank_cap_scores"])
        if diag["full_frame_rank_cap_score"] is not None:
            full_frame_rank_cap_scores.append(diag["full_frame_rank_cap_score"])
        for pred in predictions:
            x1, y1, x2, y2 = pred["xyxy"]
            detections.append({
                "image_id": image_id,
                "category_id": pred["cls"] + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(pred["score"], 5),
            })
        if (image_id + 1) % 50 == 0 or image_id + 1 == len(images):
            print(f"   tiled {image_id + 1}/{len(images)} images")

    # The final count is recoverable from the COCO list per image; compute it
    # explicitly rather than retaining another large per-image structure.
    final_by_image = defaultdict(int)
    scores_by_image = defaultdict(list)
    for det in detections:
        final_by_image[det["image_id"]] += 1
        scores_by_image[det["image_id"]].append(det["score"])
    saturation = sum(v >= tiling["global_max_det"] for v in final_by_image.values())
    diagnostics = {
        "mode": "tiled",
        "tiling": tiling,
        "images": len(images),
        "total_tiles": sum(tile_counts),
        "tiles_per_image_mean": round(statistics.fmean(tile_counts), 3),
        "tiles_per_image_min": min(tile_counts),
        "tiles_per_image_max": max(tile_counts),
        "model_forwards": sum(model_forward_counts),
        "model_forwards_per_image_mean": round(statistics.fmean(model_forward_counts), 3),
        "full_frame_views": len(images) if tiling["include_full_image"] else 0,
        "full_frame_candidates": totals["full_frame_candidates"],
        "full_frame_views_at_local_max_det": len(full_frame_rank_cap_scores),
        "full_frame_views_at_local_max_det_conf_010": sum(
            score >= 0.10 for score in full_frame_rank_cap_scores
        ),
        "full_frame_views_at_local_max_det_conf_025": sum(
            score >= 0.25 for score in full_frame_rank_cap_scores
        ),
        "full_frame_rank_max_score_median": (
            round(statistics.median(full_frame_rank_cap_scores), 5)
            if full_frame_rank_cap_scores else None
        ),
        "final_tile_detections": totals["final_tile_detections"],
        "final_full_frame_detections": totals["final_full_frame_detections"],
        "images_with_full_frame_survivor": images_with_full_frame_survivor,
        "pre_merge_detections": totals["pre_merge"],
        "post_merge_detections": totals["post_merge"],
        "final_detections": totals["final"],
        "border_candidates": totals["border_candidates"],
        "rejected_invalid_candidates": totals["rejected_invalid_candidates"],
        "invalid_final_boxes": totals["invalid_final_boxes"],
        "tiles_at_local_max_det": len(local_rank_cap_scores),
        "pct_tiles_at_local_max_det": round(100 * len(local_rank_cap_scores) / sum(tile_counts), 2),
        "tiles_at_local_max_det_conf_010": sum(score >= 0.10 for score in local_rank_cap_scores),
        "tiles_at_local_max_det_conf_025": sum(score >= 0.25 for score in local_rank_cap_scores),
        "local_rank_max_score_median": (
            round(statistics.median(local_rank_cap_scores), 5)
            if local_rank_cap_scores else None
        ),
        "images_at_global_max_det": saturation,
        "pct_images_at_global_max_det": round(100 * saturation / len(images), 2),
    }
    rank_cap_scores = [
        min(scores_by_image[image_id])
        for image_id, count in final_by_image.items()
        if count >= tiling["global_max_det"]
    ]
    diagnostics["images_at_global_max_det_conf_010"] = sum(score >= 0.10 for score in rank_cap_scores)
    diagnostics["images_at_global_max_det_conf_025"] = sum(score >= 0.25 for score in rank_cap_scores)
    diagnostics["rank_max_score_median"] = (
        round(statistics.median(rank_cap_scores), 5) if rank_cap_scores else None
    )
    diagnostics["tile_layout_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(detections))
    manifest_json.write_text(json.dumps({"tiling": tiling, "images": manifest}, indent=2) + "\n")
    diagnostics_json.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        f"wrote {len(detections)} tiled detections from {sum(tile_counts)} tiles "
        f"and {len(images) if tiling['include_full_image'] else 0} full-frame views "
        f"over {len(images)} images -> {out_json.name}"
    )
    return detections, diagnostics


def measure_tiled_speed(model, dataset: str, split: str, proto: dict, tiling: dict) -> dict:
    import torch

    from .data_coco import list_images

    latency = proto.get("latency", {})
    warmup, timed = int(latency.get("warmup", 20)), int(latency.get("timed", 100))
    images = list_images(dataset, split)
    warmup_seq = (images * (warmup // len(images) + 1))[:warmup]
    if timed == 1:
        timed_seq = [images[len(images) // 2]]
    else:
        # Tiling cost depends strongly on native dimensions. Sample the whole
        # sorted split rather than timing only its first 100 filenames.
        timed_seq = [images[round(i * (len(images) - 1) / (timed - 1))] for i in range(timed)]
    seq = warmup_seq + timed_seq
    times = []
    tile_counts = []
    model_forward_counts = []
    for index, image_path in enumerate(seq):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _, diag = _predict_image_tiled(model, image_path, proto, tiling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if index >= warmup:
            times.append((time.perf_counter() - start) * 1000)
            tile_counts.append(len(diag["tiles"]))
            model_forward_counts.append(int(diag["model_forwards"]))
    ordered = sorted(times)
    mean = statistics.fmean(ordered)
    return {
        "latency_ms_mean": round(mean, 2),
        "latency_ms_p50": round(ordered[len(ordered) // 2], 2),
        "latency_ms_p95": round(ordered[int(len(ordered) * 0.95) - 1], 2),
        "fps_batch1": round(1000 / mean, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_timed": len(ordered),
        "tiles_per_image_mean_timed": round(statistics.fmean(tile_counts), 3),
        "model_forwards_per_image_mean_timed": round(
            statistics.fmean(model_forward_counts), 3
        ),
        "timing_scope": "decode+slicing+all model views+global merge",
        "timing_sample": "100 evenly spaced images across the sorted validation split",
    }


def _iou_xywh(box: list[float], boxes: list[list[float]]) -> list[float]:
    x1, y1, w1, h1 = box
    out = []
    for x2, y2, w2, h2 in boxes:
        iw = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
        ih = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        out.append(inter / union if union > 0 else 0.0)
    return out


def native_size_recall(gt_json: Path, detections: list[dict], iou_threshold: float = 0.5) -> dict:
    """Class-aware, score-greedy recall by native GT shortest-side bin."""
    gt = json.loads(gt_json.read_text())
    gt_by_key = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_key[(ann["image_id"], ann["category_id"])].append(ann)
    det_by_key = defaultdict(list)
    for det in detections:
        det_by_key[(det["image_id"], det["category_id"])].append(det)

    matched: set[int] = set()
    for key, anns in gt_by_key.items():
        used = [False] * len(anns)
        boxes = [a["bbox"] for a in anns]
        for det in sorted(det_by_key.get(key, []), key=lambda d: -d["score"]):
            ious = _iou_xywh(det["bbox"], boxes)
            available = [i for i, value in enumerate(ious) if not used[i] and value >= iou_threshold]
            if not available:
                continue
            best = max(available, key=lambda i: ious[i])
            used[best] = True
            matched.add(int(anns[best]["id"]))

    bins = {
        "lt8": lambda value: value < 8,
        "8_to_15": lambda value: 8 <= value < 16,
        "16_to_31": lambda value: 16 <= value < 32,
        "ge32": lambda value: value >= 32,
    }
    by_size = {}
    for name, predicate in bins.items():
        anns = [a for a in gt["annotations"] if predicate(min(a["bbox"][2], a["bbox"][3]))]
        hits = sum(int(a["id"]) in matched for a in anns)
        by_size[name] = {
            "gt": len(anns),
            "matched": hits,
            "unmatched": len(anns) - hits,
            "recall": round(hits / len(anns), 5) if anns else None,
        }
    return {
        "definition": "native-source GT shortest side; same-class score-greedy IoU match",
        "iou_threshold": iou_threshold,
        "overall": {
            "gt": len(gt["annotations"]),
            "matched": len(matched),
            "unmatched": len(gt["annotations"]) - len(matched),
            "recall": round(len(matched) / len(gt["annotations"]), 5),
        },
        "by_native_short_side": by_size,
    }
