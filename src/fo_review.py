"""FiftyOne failure review: persistent dataset, per-experiment predictions,
COCO-style evaluation, saved failure views, and gate-evidence stats export.

Workflow: `review EXP --launch` opens the app (localhost:5151, reachable from the
Windows browser under WSL2). Tag failure boxes in-app with condition tags
(tiny/crowded/occluded/blur/low-light/truncation — and 'reviewed' when done),
then `review EXP --export-stats` writes artifacts/fiftyone_review.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import paths
from .data_coco import build_gt, class_names, image_dir
from .expmeta import load_config, load_protocol

CONDITION_TAGS = ("tiny", "crowded", "occluded", "blur", "low-light", "truncation", "reviewed")


def _dataset_name(dataset: str, split: str) -> str:
    return f"{dataset}-{split}"


def ensure_fo_dataset(dataset: str, split: str):
    import fiftyone as fo

    name = _dataset_name(dataset, split)
    if name in fo.list_datasets():
        return fo.load_dataset(name)
    print(f"importing {name} into FiftyOne (one-time)")
    ds = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path=str(paths.dataset_yaml(dataset)),
        split=split,
        label_field="ground_truth",
        include_all_data=True,  # keep label-less images so FP counts match COCOeval
        name=name,
    )
    ds.persistent = True
    return ds


def attach_predictions(ds, exp_dir: Path, dataset: str, split: str, seed: int) -> str:
    """Attach the experiment's COCO dets json as a label field; returns field name."""
    import fiftyone as fo

    dets_json = exp_dir / "artifacts" / f"predictions_{split}_s{seed}.json"
    if not dets_json.exists():
        raise SystemExit(f"no {dets_json.name} — run eval first")
    gt = json.loads(build_gt(dataset, split).read_text())
    dets = json.loads(dets_json.read_text())
    names = class_names(dataset)

    by_image = defaultdict(list)
    for d in dets:
        by_image[d["image_id"]].append(d)
    img_meta = {im["id"]: im for im in gt["images"]}
    dets_by_basename = {
        img_meta[i]["file_name"]: (img_meta[i], ds_list) for i, ds_list in by_image.items()
    }

    field = f"pred_{exp_dir.name}_s{seed}"
    n_boxes = 0
    for sample in ds.iter_samples(autosave=True, progress=True):
        entry = dets_by_basename.get(Path(sample.filepath).name)
        detections = []
        if entry:
            im, img_dets = entry
            W, H = im["width"], im["height"]
            for d in img_dets:
                x, y, w, h = d["bbox"]
                detections.append(fo.Detection(
                    label=names[d["category_id"] - 1],
                    bounding_box=[x / W, y / H, w / W, h / H],
                    confidence=d["score"],
                ))
            n_boxes += len(detections)
        sample[field] = fo.Detections(detections=detections)
    print(f"attached {n_boxes} predictions -> field '{field}'")
    return field


def review(exp_ref: str, seed: int | None = None, launch: bool = False,
           export_stats: bool = False) -> None:
    import fiftyone as fo
    from fiftyone import ViewField as F

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    dataset = cfg["meta"]["dataset"]
    split = load_protocol()["split"]
    if seed is None:
        preds = sorted(exp_dir.glob(f"artifacts/predictions_{split}_s*.json"))
        if not preds:
            raise SystemExit(f"{exp_dir.name}: no predictions — run eval first")
        seed = int(preds[0].stem.rsplit("_s", 1)[1])

    ds = ensure_fo_dataset(dataset, split)
    # seed-scoped identity: without it, `review --seed 42` after reviewing seed 17
    # would silently show and count seed 17's boxes
    field = f"pred_{exp_dir.name}_s{seed}"
    eval_key = f"eval_{exp_dir.name}_s{seed}"
    if field not in ds.get_field_schema():
        attach_predictions(ds, exp_dir, dataset, split, seed)
    if eval_key not in ds.list_evaluations():
        print(f"evaluating '{field}' vs ground_truth (eval_key={eval_key})")
        ds.evaluate_detections(field, gt_field="ground_truth", eval_key=eval_key,
                               compute_mAP=True)

    # saved views for the failure-analysis conditions
    tiny_thresh = (32 / load_protocol()["imgsz"]) ** 2
    views = {
        "tiny_gt": ds.filter_labels(
            "ground_truth", (F("bounding_box")[2] * F("bounding_box")[3]) < tiny_thresh),
        "crowded": ds.match(F("ground_truth.detections").length() > 30),
        f"missed_{exp_dir.name}_s{seed}": ds.filter_labels("ground_truth", F(eval_key) == "fn"),
        f"fp_{exp_dir.name}_s{seed}": ds.filter_labels(field, F(eval_key) == "fp"),
    }
    for vname, view in views.items():
        ds.save_view(vname, view, overwrite=True)

    if export_stats:
        counts = {
            "tp": ds.count_values(f"{field}.detections.{eval_key}").get("tp", 0),
            "fp": ds.count_values(f"{field}.detections.{eval_key}").get("fp", 0),
            "fn": ds.count_values(f"ground_truth.detections.{eval_key}").get("fn", 0),
        }
        fp_reviewed = ds.filter_labels(
            field, (F(eval_key) == "fp") & (F("tags").length() > 0)
        ).count(f"{field}.detections")
        fn_reviewed = ds.filter_labels(
            "ground_truth", (F(eval_key) == "fn") & (F("tags").length() > 0)
        ).count("ground_truth.detections")
        stats = {
            "dataset": ds.name,
            "eval_key": eval_key,
            "pred_field": field,
            "seed": seed,
            "counts": counts,
            "reviewed_failures": fp_reviewed + fn_reviewed,
            "label_tags": ds.count_label_tags(),
            "sample_tags": ds.count_sample_tags(),
            "saved_views": sorted(views),
            "condition_tags_expected": list(CONDITION_TAGS),
        }
        out = exp_dir / "artifacts" / "fiftyone_review.json"
        out.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {out.relative_to(paths.ROOT)}  "
              f"(tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} "
              f"reviewed={stats['reviewed_failures']})")

    if launch:
        print("launching FiftyOne app — open http://localhost:5151 in the Windows browser")
        print(f"tag failure boxes with: {', '.join(CONDITION_TAGS)}")
        session = fo.launch_app(ds, port=5151)
        session.wait()
