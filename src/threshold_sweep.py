"""Post-hoc confidence calibration from fixed evaluation predictions.

This module never runs the detector. It filters an existing COCO prediction
artifact at each requested confidence threshold, performs class-aware greedy
one-to-one matching, and writes a reproducible JSON + Markdown report.

The result is an operating-point calibration of one frozen prediction set. It
is not a model improvement and must not be reported as retraining evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import paths
from .data_coco import build_gt, gt_fingerprint
from .expmeta import (append_command_sh, effective_protocol_hash, file_sha16,
                      load_config, load_protocol, protocol_overrides)


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
CALIBRATION_NOTE = (
    "This sweep only calibrates confidence thresholds on a frozen, post-NMS, "
    "post-max_det prediction artifact. It is not a model improvement, does not "
    "recover suppressed detections, and is tuned on this labeled split."
)
LIMITATIONS = [
    "The prediction artifact is already post-NMS and post-max_det, so thresholding cannot recover suppressed or truncated boxes.",
    "The selected threshold is tuned and measured on the same validation split; confirm it on untouched labeled data before deployment claims.",
    "F1 depends on the configured class-aware IoU criterion and is not COCO mAP or a replacement for the fixed-protocol metrics.",
    "The harness GT has no COCO crowd/ignore annotations; omitted VisDrone ignore regions can inflate apparent false positives.",
    "Per-class threshold selection has greater overfitting risk than one global threshold and requires held-out confirmation.",
    "Prediction scores are stored at finite precision in the evaluation artifact, so equality follows the serialized values.",
]


def _xywh_iou(box: list[float], boxes: np.ndarray) -> np.ndarray:
    """Return IoU between one xywh box and an ``[N, 4]`` xywh array."""
    if boxes.size == 0:
        return np.zeros(0, dtype=float)
    x, y, w, h = (float(v) for v in box)
    x2, y2 = x + w, y + h
    ix = np.maximum(0.0, np.minimum(x2, boxes[:, 0] + boxes[:, 2]) - np.maximum(x, boxes[:, 0]))
    iy = np.maximum(0.0, np.minimum(y2, boxes[:, 1] + boxes[:, 3]) - np.maximum(y, boxes[:, 1]))
    intersection = ix * iy
    union = w * h + boxes[:, 2] * boxes[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _prepare(gt: dict, predictions: list[dict]) -> tuple[list[int], dict, int]:
    """Precompute score-sorted predictions and their same-class GT IoUs."""
    class_ids = [int(c["id"]) for c in gt["categories"]]
    names = {int(c["id"]): str(c["name"]) for c in gt["categories"]}
    known_images = {int(im["id"]) for im in gt["images"]}
    gt_by_key: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    pred_by_key: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)

    if any(int(a.get("iscrowd", 0)) for a in gt["annotations"]):
        raise SystemExit(
            "threshold sweep currently requires non-crowd harness GT; "
            "ignore/crowd matching needs separate semantics"
        )
    for ann in gt["annotations"]:
        key = (int(ann["image_id"]), int(ann["category_id"]))
        if key[0] not in known_images or key[1] not in names:
            raise SystemExit(f"invalid GT image/category reference: {key}")
        gt_by_key[key].append([float(v) for v in ann["bbox"]])

    for order, pred in enumerate(predictions):
        key = (int(pred["image_id"]), int(pred["category_id"]))
        score = float(pred["score"])
        if key[0] not in known_images or key[1] not in names:
            raise SystemExit(f"invalid prediction image/category reference: {key}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise SystemExit(f"invalid prediction score at index {order}: {score}")
        pred_by_key[key].append((order, pred))

    records = {}
    for image_id in known_images:
        for class_id in class_ids:
            key = (image_id, class_id)
            gt_boxes = np.asarray(gt_by_key.get(key, []), dtype=float).reshape(-1, 4)
            ordered = sorted(
                pred_by_key.get(key, []),
                key=lambda item: (-float(item[1]["score"]), item[0]),
            )
            scores = np.asarray([float(p["score"]) for _, p in ordered], dtype=float)
            if ordered and len(gt_boxes):
                ious = np.stack(
                    [_xywh_iou([float(v) for v in p["bbox"]], gt_boxes) for _, p in ordered]
                )
            else:
                ious = np.zeros((len(ordered), len(gt_boxes)), dtype=float)
            records[key] = {"gt": len(gt_boxes), "scores": scores, "ious": ious}

    return class_ids, {"names": names, "records": records}, len(known_images)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _evaluate_prepared(
    class_ids: list[int], prepared: dict, n_images: int, threshold: float, iou_threshold: float
) -> dict:
    """Evaluate one confidence threshold with score-ordered same-class matching."""
    rows = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "predictions": 0, "gt": 0}
    for class_id in class_ids:
        counts = {"tp": 0, "fp": 0, "fn": 0, "predictions": 0, "gt": 0}
        for (image_id, cid), record in prepared["records"].items():
            if cid != class_id:
                continue
            n_gt = int(record["gt"])
            matched = np.zeros(n_gt, dtype=bool)
            keep = np.flatnonzero(record["scores"] >= threshold)
            counts["gt"] += n_gt
            counts["predictions"] += len(keep)
            for pred_idx in keep:
                available = np.flatnonzero(~matched)
                if len(available) == 0:
                    counts["fp"] += 1
                    continue
                candidate_ious = record["ious"][pred_idx, available]
                best_local = int(np.argmax(candidate_ious))
                if float(candidate_ious[best_local]) >= iou_threshold:
                    matched[available[best_local]] = True
                    counts["tp"] += 1
                else:
                    counts["fp"] += 1
            counts["fn"] += n_gt - int(matched.sum())

        metrics = _prf(counts["tp"], counts["fp"], counts["fn"])
        rows[prepared["names"][class_id]] = {**counts, **metrics}
        for key in totals:
            totals[key] += counts[key]

    micro = {**totals, **_prf(totals["tp"], totals["fp"], totals["fn"])}
    macro = {
        key: round(float(np.mean([row[key] for row in rows.values()])), 6)
        for key in ("precision", "recall", "f1")
    }
    return {
        "threshold": threshold,
        "overall": {
            "micro": micro,
            "macro": macro,
            "fp_per_image": round(totals["fp"] / n_images, 6),
            "unmatched_gt": totals["fn"],
            "unmatched_gt_pct": round(100 * totals["fn"] / totals["gt"], 4) if totals["gt"] else 0.0,
        },
        "per_class": rows,
    }


def sweep_records(
    gt: dict,
    predictions: list[dict],
    thresholds: list[float] | tuple[float, ...],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Pure, testable threshold-sweep core over COCO-style dictionaries."""
    class_ids, prepared, n_images = _prepare(gt, predictions)
    return [
        _evaluate_prepared(class_ids, prepared, n_images, threshold, iou_threshold)
        for threshold in thresholds
    ]


def _best(rows: list[dict], metric_path: tuple[str, ...]) -> dict:
    """Maximize a metric; exact ties choose the higher threshold (fewer boxes)."""
    def value(row: dict) -> float:
        cur = row
        for key in metric_path:
            cur = cur[key]
        return float(cur)

    winner = max(rows, key=lambda row: (value(row), float(row["threshold"])))
    return {"threshold": winner["threshold"], "value": value(winner)}


def select_best(rows: list[dict]) -> dict:
    """Return macro/micro global optima and one F1 optimum per class."""
    classes = list(rows[0]["per_class"])
    per_class = {}
    for name in classes:
        winner = max(
            rows,
            key=lambda row: (row["per_class"][name]["f1"], float(row["threshold"])),
        )
        per_class[name] = {
            "threshold": winner["threshold"],
            "at_candidate_grid_boundary": winner["threshold"] in {
                rows[0]["threshold"], rows[-1]["threshold"]
            },
            **{k: winner["per_class"][name][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
        }
    macro = _best(rows, ("overall", "macro", "f1"))
    micro = _best(rows, ("overall", "micro", "f1"))
    boundaries = {rows[0]["threshold"], rows[-1]["threshold"]}
    return {
        "candidate_grid_only": True,
        "recommended_global": {
            **macro,
            "metric": "macro_f1",
            "tie_break": "higher confidence threshold",
            "at_candidate_grid_boundary": macro["threshold"] in boundaries,
        },
        "best_micro_f1": {
            **micro,
            "metric": "micro_f1",
            "at_candidate_grid_boundary": micro["threshold"] in boundaries,
        },
        "best_per_class": per_class,
    }


def _prediction_file(exp_dir: Path, split: str, seed: int | None) -> tuple[int, Path]:
    if seed is not None:
        path = exp_dir / "artifacts" / f"predictions_{split}_s{seed}.json"
        if not path.exists():
            raise SystemExit(f"{exp_dir.name}: no {path.name} — run eval first")
        return seed, path
    candidates = sorted(exp_dir.glob(f"artifacts/predictions_{split}_s*.json"))
    if not candidates:
        raise SystemExit(f"{exp_dir.name}: no predictions for split '{split}' — run eval first")
    if len(candidates) > 1:
        raise SystemExit(f"{exp_dir.name}: multiple prediction seeds found — pass --seed")
    match = re.search(r"_s(\d+)$", candidates[0].stem)
    if not match:
        raise SystemExit(f"cannot parse seed from {candidates[0].name}")
    return int(match.group(1)), candidates[0]


def _markdown(report: dict) -> str:
    p = report["provenance"]
    lines = [
        f"# Threshold sweep: {p['experiment']} seed {p['seed']}",
        "",
        f"> {report['note']}",
        "",
        "## Provenance and matching",
        "",
        f"- Dataset: `{p['dataset']}/{p['split']}`; GT fingerprint `{p['ground_truth']['fingerprint']}`",
        f"- Protocol: `{p['protocol']['hash']}`; checkpoint `{p['checkpoint']['sha256']}`",
        f"- Predictions: `{p['predictions']['file']}` (`{p['predictions']['sha256']}`, {p['predictions']['count']:,} boxes)",
        f"- Matching: class-aware, prediction-score-descending greedy one-to-one at IoU >= {report['matching']['iou_threshold']:.2f}",
        "- Threshold inclusion: `score >= threshold`; equal-score ties retain artifact order",
        "",
        "## Overall sweep",
        "",
        "| conf | micro P | micro R | micro F1 | macro P | macro R | macro F1 | FP/image | unmatched GT |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        mi, ma, overall = row["overall"]["micro"], row["overall"]["macro"], row["overall"]
        lines.append(
            f"| {row['threshold']:.2f} | {mi['precision']:.4f} | {mi['recall']:.4f} | "
            f"{mi['f1']:.4f} | {ma['precision']:.4f} | {ma['recall']:.4f} | "
            f"{ma['f1']:.4f} | {overall['fp_per_image']:.2f} | {overall['unmatched_gt']:,} |"
        )

    selected = report["selection"]
    lines += [
        "",
        "## Selected thresholds",
        "",
        f"Recommended global threshold from this candidate grid: **{selected['recommended_global']['threshold']:.2f}** "
        f"(maximum macro F1 {selected['recommended_global']['value']:.4f}; exact ties choose the higher threshold).",
        "",
        f"Micro-F1 optimum: **{selected['best_micro_f1']['threshold']:.2f}** "
        f"({selected['best_micro_f1']['value']:.4f}).",
    ]
    if selected["recommended_global"]["at_candidate_grid_boundary"]:
        lines += [
            "",
            "> The macro-F1 winner is on the candidate-grid boundary. Expand the grid before treating it as the true optimum.",
        ]
    lines += [
        "",
        "| class | best conf | precision | recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in selected["best_per_class"].items():
        lines.append(
            f"| {name} | {row['threshold']:.2f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} |"
        )

    lines += ["", "## Per-class results", ""]
    for result in report["results"]:
        lines += [
            f"### Confidence {result['threshold']:.2f}",
            "",
            "| class | precision | recall | F1 | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in result["per_class"].items():
            lines.append(
                f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['tp']:,} | {row['fp']:,} | {row['fn']:,} |"
            )
        lines.append("")
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def threshold_sweep(
    exp_ref: str,
    seed: int | None = None,
    thresholds: list[float] | None = None,
    iou_threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Run the fixed-artifact confidence sweep and write JSON + Markdown."""
    if not 0.0 < iou_threshold <= 1.0 or not math.isfinite(iou_threshold):
        raise SystemExit("--iou must be finite and in (0, 1]")
    values = list(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    if not values:
        raise SystemExit("at least one confidence threshold is required")
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise SystemExit("all confidence thresholds must be finite and in [0, 1]")
    values = sorted(set(float(v) for v in values))

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    dataset = cfg["meta"]["dataset"]
    overrides = protocol_overrides(cfg)
    proto = load_protocol(overrides)
    split = proto["split"]
    seed, pred_path = _prediction_file(exp_dir, split, seed)
    metrics_path = exp_dir / "seeds" / f"s{seed}" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"{exp_dir.name}: no seed-{seed} metrics.json — run eval first")
    metrics = json.loads(metrics_path.read_text())
    expected_protocol = effective_protocol_hash(cfg)
    if metrics.get("protocol", {}).get("hash") != expected_protocol:
        raise SystemExit(f"{exp_dir.name}: seed-{seed} metrics use a stale protocol — run eval again")

    gt_path = build_gt(dataset, split)
    gt_fp = gt_fingerprint(dataset, split)
    if metrics.get("protocol", {}).get("gt") != gt_fp:
        raise SystemExit(f"{exp_dir.name}: seed-{seed} metrics use stale ground truth — run eval again")
    gt = json.loads(gt_path.read_text())
    predictions = json.loads(pred_path.read_text())
    if not isinstance(predictions, list):
        raise SystemExit(f"{pred_path.name}: expected a JSON list of detections")

    recorded_weights = metrics.get("model", {}).get("weights_sha256")
    weights_path = Path(metrics.get("model", {}).get("weights", ""))
    if not recorded_weights or not weights_path.is_file():
        raise SystemExit(f"{exp_dir.name}: missing checkpoint provenance in seed metrics")
    actual_weights = file_sha16(weights_path)
    if actual_weights != recorded_weights:
        raise SystemExit(
            f"{exp_dir.name}: checkpoint changed since eval "
            f"({recorded_weights} -> {actual_weights}) — re-run eval first"
        )

    rows = sweep_records(gt, predictions, values, iou_threshold=iou_threshold)
    report = {
        "method": "posthoc_confidence_threshold_sweep",
        "note": CALIBRATION_NOTE,
        "limitations": LIMITATIONS,
        "provenance": {
            "experiment": exp_dir.name,
            "seed": seed,
            "dataset": dataset,
            "split": split,
            "protocol": metrics["protocol"],
            "checkpoint": {"path": str(weights_path), "sha256": actual_weights},
            "predictions": {
                "file": pred_path.name,
                "path": str(pred_path),
                "sha256": file_sha16(pred_path),
                "count": len(predictions),
                "artifact_confidence_floor": proto["conf"],
                "nms_iou": proto["iou"],
                "max_det": proto["max_det"],
            },
            "ground_truth": {
                "path": str(gt_path),
                "sha256": file_sha16(gt_path),
                "fingerprint": gt_fp,
                "images": len(gt["images"]),
                "annotations": len(gt["annotations"]),
            },
        },
        "matching": {
            "class_aware": True,
            "one_to_one": True,
            "order": "prediction score descending; artifact order breaks equal-score ties",
            "iou_threshold": iou_threshold,
            "iou_comparison": ">=",
            "score_comparison": ">=",
            "crowd_or_ignore_support": False,
        },
        "thresholds": values,
        "selection": select_best(rows),
        "results": rows,
    }

    tag = f"threshold_sweep_{split}_s{seed}_iou{round(iou_threshold * 100):02d}"
    json_path = exp_dir / "artifacts" / f"{tag}.json"
    md_path = exp_dir / "artifacts" / f"{tag}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md = _markdown(report)
    md_path.write_text(md)

    argv = ["threshold-sweep", exp_dir.name, "--seed", str(seed), "--iou", str(iou_threshold),
            "--thresholds", ",".join(str(v) for v in values)]
    append_command_sh(exp_dir, argv)
    print(md)
    print(f"wrote {json_path.relative_to(paths.ROOT)}")
    print(f"wrote {md_path.relative_to(paths.ROOT)}")
    return json_path, md_path
