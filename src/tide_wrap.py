"""TIDE error taxonomy -> tide_report.json.

Primary: tidecv (2020-era; needs a numpy attribute shim). If it fails for any
reason, an internal greedy matcher computes TIDE-style error *counts* (same
thresholds tb=0.1, tf=0.5) so downstream consumers always get the same file
shape — `method` says which path produced it."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import paths
from .data_coco import build_gt
from .expmeta import load_config, load_protocol

TB, TF = 0.1, 0.5  # TIDE background / foreground IoU thresholds


def _numpy_shim() -> None:
    import numpy as np
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)


def _run_tidecv(gt_json: Path, dets_json: Path, max_det: int) -> dict:
    # Build tidecv Data objects directly: its own COCO loader requires a
    # 'segmentation' field on every annotation and hardcodes max_dets=100,
    # which would diverge from our protocol.
    _numpy_shim()
    from tidecv import TIDE
    from tidecv.data import Data

    gt = json.loads(gt_json.read_text())
    gt_data = Data("gt", max_dets=max_det)
    for im in gt["images"]:
        gt_data.add_image(im["id"], im["file_name"])
    for c in gt["categories"]:
        gt_data.add_class(c["id"], c["name"])
    for a in gt["annotations"]:
        gt_data.add_ground_truth(a["image_id"], a["category_id"], a["bbox"])

    det_data = Data("preds", max_dets=max_det)
    for d in json.loads(dets_json.read_text()):
        det_data.add_detection(d["image_id"], d["category_id"], d["score"], d["bbox"])

    tide = TIDE()
    tide.evaluate_range(gt_data, det_data, mode=TIDE.BOX, name="run")
    main = tide.get_main_errors()["run"]
    special = tide.get_special_errors()["run"]
    return {
        "method": "tidecv",
        "dAP": {k: round(float(v), 3) for k, v in main.items()},
        "special_dAP": {k: round(float(v), 3) for k, v in special.items()},
        "counts": None,
        "notes": "dAP = AP gained by oracle-fixing that error type (TIDE paper definition)",
    }


def _iou(box: list[float], others: list[list[float]]) -> list[float]:
    """IoU of one xywh box vs a list of xywh boxes."""
    x1, y1, w1, h1 = box
    out = []
    for x2, y2, w2, h2 in others:
        ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = ix * iy
        union = w1 * h1 + w2 * h2 - inter
        out.append(inter / union if union > 0 else 0.0)
    return out


def _fallback_counts(gt_json: Path, dets_json: Path) -> dict:
    gt = json.loads(gt_json.read_text())
    dets = json.loads(dets_json.read_text())

    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    dets_by_img = defaultdict(list)
    for d in dets:
        dets_by_img[d["image_id"]].append(d)

    counts = {"Cls": 0, "Loc": 0, "Both": 0, "Dupe": 0, "Bkg": 0, "Miss": 0, "TP": 0}
    for img_id, img_dets in dets_by_img.items():
        anns = gt_by_img.get(img_id, [])
        boxes = [a["bbox"] for a in anns]
        used = [False] * len(anns)
        for d in sorted(img_dets, key=lambda x: -x["score"]):
            ious = _iou(d["bbox"], boxes)
            same = [(i, v) for i, v in enumerate(ious) if anns[i]["category_id"] == d["category_id"]]
            diff = [(i, v) for i, v in enumerate(ious) if anns[i]["category_id"] != d["category_id"]]
            best_same_free = max((v for i, v in same if not used[i]), default=0.0)
            best_same_used = max((v for i, v in same if used[i]), default=0.0)
            best_diff = max((v for _, v in diff), default=0.0)
            if best_same_free >= TF:
                idx = max((i for i, v in same if not used[i]), key=lambda i: ious[i])
                used[idx] = True
                counts["TP"] += 1
            elif best_same_used >= TF:
                counts["Dupe"] += 1
            elif best_diff >= TF:
                counts["Cls"] += 1
            elif best_same_free >= TB or best_same_used >= TB:
                counts["Loc"] += 1
            elif best_diff >= TB:
                counts["Both"] += 1
            else:
                counts["Bkg"] += 1
    total_gt = sum(len(v) for v in gt_by_img.values())
    counts["Miss"] = max(0, total_gt - counts["TP"])
    return {
        "method": "greedy-count-fallback",
        "dAP": None,
        "special_dAP": None,
        "counts": counts,
        "notes": f"tidecv unavailable; greedy per-image matching with tb={TB}, tf={TF}. "
                 "Counts (not dAP): Miss counts every unmatched GT, incl. those behind Loc/Cls errors.",
    }


def run_tide(exp_ref: str, seed: int | None = None) -> None:
    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    dataset = cfg["meta"]["dataset"]
    split = load_protocol()["split"]
    gt_json = build_gt(dataset, split)

    preds = sorted(exp_dir.glob(f"artifacts/predictions_{split}_s*.json"))
    if seed is not None:
        preds = [p for p in preds if p.stem.endswith(f"_s{seed}")]
    if not preds:
        raise SystemExit(f"{exp_dir.name}: no predictions for split '{split}' — run eval first")
    dets_json = preds[0]

    try:
        report = _run_tidecv(gt_json, dets_json, load_protocol()["max_det"])
    except Exception as e:  # stale tidecv is a known risk — fall back, don't die
        print(f"tidecv failed ({type(e).__name__}: {e}); using greedy-count fallback")
        report = _fallback_counts(gt_json, dets_json)

    from .expmeta import append_command_sh, file_sha16
    append_command_sh(exp_dir, ["tide", exp_dir.name]
                      + (["--seed", str(seed)] if seed is not None else []))

    report["dataset"] = dataset
    report["split"] = split
    report["predictions"] = dets_json.name
    report["predictions_sha256"] = file_sha16(dets_json)
    out = exp_dir / "tide_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out.relative_to(paths.ROOT)} (method={report['method']})")
    if report["dAP"]:
        print("   dAP:", ", ".join(f"{k}={v}" for k, v in report["dAP"].items()))
    elif report["counts"]:
        print("   counts:", ", ".join(f"{k}={v}" for k, v in report["counts"].items()))
