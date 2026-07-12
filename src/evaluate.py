"""Fixed-protocol evaluation -> metrics.json.

Accuracy: our own predict loop -> COCO detections JSON -> pycocotools COCOeval
(overall + per-class + small/medium/large). Speed: a separate batch=1 timed pass.
Every metrics.json records the protocol hash; compare refuses mismatches."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from . import paths
from .data_coco import build_gt, class_names, image_dir, list_images
from .expmeta import (append_command_sh, env_versions, file_sha16, load_config,
                      load_protocol, protocol_hash)

AGG_KEYS = ("map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large",
            "latency_ms_mean", "fps_batch1")


def _weights_for_seed(exp_dir: Path, seed: int) -> Path | None:
    w = exp_dir / "seeds" / f"s{seed}" / "weights" / "best.pt"
    return w if w.exists() else None


def _seeds_with_weights(exp_dir: Path) -> list[int]:
    return sorted(
        int(d.name[1:]) for d in (exp_dir / "seeds").glob("s*")
        if (d / "weights" / "best.pt").exists()
    )


def predict_to_coco(model, dataset: str, split: str, proto: dict, out_json: Path) -> list[dict]:
    """Run the model over the split, write a COCO detections json (image order == GT ids)."""
    imgs = list_images(dataset, split)
    dets = []
    results = model.predict(
        [str(p) for p in imgs],
        imgsz=proto["imgsz"], conf=proto["conf"], iou=proto["iou"],
        max_det=proto["max_det"], device=proto["device"],
        stream=True, verbose=False,
    )
    for img_id, r in enumerate(results):
        for xyxy, score, cls in zip(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        ):
            x1, y1, x2, y2 = xyxy
            dets.append({
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(score, 5),
            })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dets))
    print(f"wrote {len(dets)} detections over {len(imgs)} images -> {out_json.name}")
    return dets


def coco_metrics(gt_json: Path, dets: list[dict], max_det: int, names: dict[int, str],
                 img_ids: list[int] | None = None) -> dict:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))
    if not dets:
        return {"overall": {k: 0.0 for k in
                            ("map50_95", "map50", "map75", "ap_small", "ap_medium",
                             "ap_large", "ar_max")},
                "per_class": {n: {"ap50_95": 0.0, "ap50": 0.0} for n in names.values()}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.maxDets = [1, 10, max_det]
        if img_ids is not None:  # subset evaluation (e.g. parity's image sample)
            e.params.imgIds = img_ids
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats
    # stats[0] is broken under custom maxDets (pycocotools' _summarize(1) uses a
    # hardcoded default maxDets=100) — compute it from the precision array instead
    pr = e.eval["precision"][:, :, :, 0, -1]
    map50_95 = float(pr[pr > -1].mean()) if (pr > -1).any() else 0.0
    overall = {
        "map50_95": round(map50_95, 5), "map50": round(float(s[1]), 5),
        "map75": round(float(s[2]), 5), "ap_small": round(float(s[3]), 5),
        "ap_medium": round(float(s[4]), 5), "ap_large": round(float(s[5]), 5),
        "ar_max": round(float(s[8]), 5),
    }
    per_class = {}
    precision = e.eval["precision"]  # [T, R, K, A, M]
    for k, cat_id in enumerate(e.params.catIds):
        name = names.get(cat_id - 1, str(cat_id))
        pr_all = precision[:, :, k, 0, -1]
        pr_50 = precision[0, :, k, 0, -1]
        per_class[name] = {
            "ap50_95": round(float(pr_all[pr_all > -1].mean()), 5) if (pr_all > -1).any() else 0.0,
            "ap50": round(float(pr_50[pr_50 > -1].mean()), 5) if (pr_50 > -1).any() else 0.0,
        }
    return {"overall": overall, "per_class": per_class}


def measure_speed(model, dataset: str, split: str, proto: dict) -> dict:
    import torch

    lat_cfg = proto.get("latency", {})
    warmup, timed = lat_cfg.get("warmup", 20), lat_cfg.get("timed", 100)
    imgs = [str(p) for p in list_images(dataset, split)]
    seq = (imgs * ((warmup + timed) // len(imgs) + 1))[: warmup + timed]

    times = []
    for i, p in enumerate(seq):
        t0 = time.perf_counter()
        model.predict(p, imgsz=proto["imgsz"], conf=proto["conf"], iou=proto["iou"],
                      max_det=proto["max_det"], device=proto["device"], verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    mean = statistics.fmean(times)
    return {
        "latency_ms_mean": round(mean, 2),
        "latency_ms_p50": round(times[len(times) // 2], 2),
        "latency_ms_p95": round(times[int(len(times) * 0.95) - 1], 2),
        "fps_batch1": round(1000 / mean, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_timed": len(times),
    }


def evaluate(exp_ref: str, seed: int | None = None, weights: str | None = None) -> None:
    from ultralytics import YOLO

    from .data_coco import gt_fingerprint

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    dataset = cfg["meta"]["dataset"]
    proto = load_protocol()
    split = proto["split"]
    names = class_names(dataset)
    gt_json = build_gt(dataset, split)
    gt_fp = gt_fingerprint(dataset, split)

    argv = ["eval", exp_dir.name]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if weights:
        argv += ["--weights", str(weights)]
    append_command_sh(exp_dir, argv)

    # (seed_label, weights_path, is_custom): ad-hoc weights never masquerade as a
    # trained seed — their metrics go to artifacts/, not seeds/, and are not aggregated
    if weights:
        entries = [("custom", Path(weights), True)]
    elif seed is not None:
        w = _weights_for_seed(exp_dir, seed)
        if not w:
            raise SystemExit(f"no weights for seed {seed} — train first")
        entries = [(seed, w, False)]
    else:
        entries = [(s, _weights_for_seed(exp_dir, s), False)
                   for s in _seeds_with_weights(exp_dir)]
        if not entries:
            raise SystemExit(f"{exp_dir.name}: no trained seeds found — train first")

    for s, w, custom in entries:
        print(f"== eval {exp_dir.name} seed {s} ({w})")
        model = YOLO(str(w))
        tag = f"custom_{w.stem}" if custom else f"s{s}"
        dets_json = exp_dir / "artifacts" / f"predictions_{split}_{tag}.json"
        dets = predict_to_coco(model, dataset, split, proto, dets_json)
        acc = coco_metrics(gt_json, dets, proto["max_det"], names)
        speed = measure_speed(model, dataset, split, proto)
        # model.info(verbose=False) returns None in this ultralytics version —
        # compute directly; None (not 0) on failure so gates flag it as missing
        try:
            from ultralytics.utils.torch_utils import get_flops, get_num_params
            n_p = get_num_params(model.model)
            flops = get_flops(model.model, proto["imgsz"])
        except Exception:
            n_p, flops = None, None
        seed_metrics = {
            "protocol": {"hash": protocol_hash(), "dataset": dataset, "split": split,
                         "gt": gt_fp, "imgsz": proto["imgsz"], "conf": proto["conf"],
                         "iou": proto["iou"], "max_det": proto["max_det"]},
            "seed": None if custom else s,
            "custom_weights": str(w) if custom else None,
            "overall": acc["overall"],
            "per_class": acc["per_class"],
            "speed": speed,
            "model": {"params_m": round(n_p / 1e6, 3) if n_p else None,
                      "gflops": round(float(flops), 2) if flops else None,
                      "weights": str(w),
                      "weights_sha256": file_sha16(w)},
            "env": env_versions(),
        }
        if custom:
            out = exp_dir / "artifacts" / f"metrics_custom_{w.stem}.json"
            print(f"   custom weights — writing {out.name}, NOT counted as a seed")
        else:
            out = paths.seed_dir(exp_dir, s) / "metrics.json"
        out.write_text(json.dumps(seed_metrics, indent=2) + "\n")
        print(f"   mAP50-95={acc['overall']['map50_95']:.4f}  mAP50={acc['overall']['map50']:.4f}  "
              f"AP_s={acc['overall']['ap_small']:.4f}  {speed['latency_ms_mean']}ms/img")

        del model
        from .train import _free_gpu
        _free_gpu()

    if not any(custom for _, _, custom in entries):
        aggregate_exp(exp_dir)


def aggregate_exp(exp_dir: Path) -> None:
    """Merge per-seed metrics (same protocol hash) into experiment-level metrics.json."""
    per_seed, newest = {}, None
    for mj in sorted(exp_dir.glob("seeds/s*/metrics.json")):
        m = json.loads(mj.read_text())
        if m["protocol"]["hash"] == protocol_hash():
            per_seed[f"s{m['seed']}"] = m
            if newest is None or mj.stat().st_mtime > newest[0]:
                newest = (mj.stat().st_mtime, m["protocol"].get("gt"))
    # never average seeds evaluated against different ground truth
    if newest and len({m["protocol"].get("gt") for m in per_seed.values()}) > 1:
        stale = [k for k, m in per_seed.items() if m["protocol"].get("gt") != newest[1]]
        print(f"WARNING: dropping {stale} from aggregation — evaluated against older GT; re-run eval")
        per_seed = {k: m for k, m in per_seed.items() if k not in stale}
    if not per_seed:
        return
    first = next(iter(per_seed.values()))

    def collect(key_path):
        vals = []
        for m in per_seed.values():
            v = m
            for k in key_path:
                v = v.get(k, {})
            if isinstance(v, (int, float)):
                vals.append(v)
        return vals

    aggregate = {}
    for key in AGG_KEYS:
        sect = "overall" if key.startswith(("map", "ap_")) else "speed"
        vals = collect([sect, key])
        if vals:
            aggregate[f"{key}_mean"] = round(statistics.fmean(vals), 5)
            aggregate[f"{key}_std"] = round(statistics.stdev(vals), 5) if len(vals) > 1 else None

    # per-class stats across ALL seeds, not just the first
    per_class = {}
    for cls in first["per_class"]:
        vals = [m["per_class"][cls]["ap50_95"] for m in per_seed.values()
                if cls in m["per_class"]]
        vals50 = [m["per_class"][cls]["ap50"] for m in per_seed.values()
                  if cls in m["per_class"]]
        per_class[cls] = {
            "ap50_95": round(statistics.fmean(vals), 5),
            "ap50": round(statistics.fmean(vals50), 5),
            "ap50_95_std": round(statistics.stdev(vals), 5) if len(vals) > 1 else None,
        }

    exp_metrics = {
        "protocol": first["protocol"],
        "n_seeds": len(per_seed),
        "aggregate": aggregate,
        "overall": first["overall"] if len(per_seed) == 1 else None,
        "per_class": per_class,
        "speed": first["speed"],
        "model": first["model"],
        "seeds": {k: {"map50_95": v["overall"]["map50_95"],
                      "map50": v["overall"]["map50"],
                      "ap_small": v["overall"]["ap_small"],
                      "weights_sha256": v["model"].get("weights_sha256")}
                  for k, v in per_seed.items()},
        "env": first["env"],
    }
    (exp_dir / "metrics.json").write_text(json.dumps(exp_metrics, indent=2) + "\n")
    print(f"aggregated {len(per_seed)} seed(s) -> {exp_dir.name}/metrics.json")
