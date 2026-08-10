"""Fixed-protocol evaluation -> metrics.json.

Accuracy: our own predict loop -> COCO detections JSON -> pycocotools COCOeval
(overall + per-class + small/medium/large). Speed: a separate batch=1 timed pass.
Every metrics.json records the protocol hash; compare refuses mismatches."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from . import paths
from .data_coco import build_gt, class_names, image_dir, list_images
from .expmeta import (append_command_sh, effective_protocol_hash, env_versions,
                      file_sha16, inference_identity, load_config, load_protocol,
                      protocol_hash, protocol_overrides)

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


def _evaluation_weights(exp_dir: Path, cfg: dict, seed: int) -> tuple[Path | None, dict | None]:
    """Resolve local weights or an explicitly inherited evaluation checkpoint."""
    local = _weights_for_seed(exp_dir, seed)
    if local:
        return local, None

    spec = (cfg.get("evaluation") or {}).get("weights_from")
    if not spec:
        return None, None
    if not isinstance(spec, dict) or not spec.get("experiment"):
        raise SystemExit("evaluation.weights_from requires an experiment")
    source_exp = paths.resolve_exp(str(spec["experiment"]))
    source_seed = int(spec.get("seed", seed))
    inherited = _weights_for_seed(source_exp, source_seed)
    if not inherited:
        raise SystemExit(
            f"no inherited weights for {source_exp.name} seed {source_seed}"
        )
    expected_sha = spec.get("sha256")
    if expected_sha and file_sha16(inherited) != str(expected_sha):
        raise SystemExit(
            f"inherited checkpoint changed: expected {expected_sha}, "
            f"got {file_sha16(inherited)} — update the experiment hypothesis or restore the checkpoint"
        )
    return inherited, {"experiment": source_exp.name, "seed": source_seed}


def predict_to_coco(model, dataset: str, split: str, proto: dict, out_json: Path) -> list[dict]:
    """Run the model over the split, write a COCO detections json (image order == GT ids)."""
    imgs = list_images(dataset, split)
    dets = []
    # A Python list of paths is treated by Ultralytics as one in-memory image
    # batch (LoadPilAndNumpy), regardless of stream=True. On VisDrone that tries
    # to place all 548 validation images on the GPU together and OOMs larger
    # architectures such as the P2-head model. A directory source uses
    # LoadImagesAndVideos and honors batch=1 while retaining sorted filename
    # order, which is also how build_gt assigns image ids. Keep rect=False
    # explicit: the former mixed-shape mega-batch also used square letterboxing,
    # and changing that here would invalidate comparisons with existing metrics.
    results = model.predict(
        str(image_dir(dataset, split)), batch=1, rect=False,
        imgsz=proto["imgsz"], conf=proto["conf"], iou=proto["iou"],
        max_det=proto["max_det"], device=proto["device"],
        stream=True, verbose=False,
    )
    for img_id, r in enumerate(results):
        expected = imgs[img_id]
        if Path(r.path).name != expected.name:
            raise RuntimeError(
                f"prediction order mismatch at image {img_id}: "
                f"expected {expected.name}, got {Path(r.path).name}"
            )
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
    from .data_coco import gt_fingerprint
    from .convnext_frcnn import is_convnext_frcnn

    exp_dir = paths.resolve_exp(exp_ref)
    cfg = load_config(exp_dir)
    custom_family = is_convnext_frcnn(cfg)
    if not custom_family:
        from ultralytics import YOLO
    dataset = cfg["meta"]["dataset"]
    overrides = protocol_overrides(cfg)
    proto = load_protocol(overrides)
    effective_hash = effective_protocol_hash(cfg)
    identity_extra = inference_identity(cfg)
    from .sliced_inference import normalized_tiling
    tiling = normalized_tiling(cfg, proto)
    split = proto["split"]
    names = class_names(dataset)
    gt_json = build_gt(dataset, split)
    gt_fp = gt_fingerprint(dataset, split)
    expected_gt = (cfg.get("evaluation") or {}).get("expected_gt")
    if expected_gt and gt_fp != str(expected_gt):
        raise SystemExit(
            f"evaluation ground truth changed: expected {expected_gt}, got {gt_fp}"
        )

    argv = ["eval", exp_dir.name]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if weights:
        argv += ["--weights", str(weights)]
    append_command_sh(exp_dir, argv)

    # (seed_label, weights_path, is_custom, inherited_source): ad-hoc weights
    # never masquerade as a trained seed. Explicit inherited weights are a
    # reproducible inference-only experiment and retain their declared seed.
    if weights:
        entries = [("custom", Path(weights), True, None)]
    elif seed is not None:
        w, source = _evaluation_weights(exp_dir, cfg, seed)
        if not w:
            raise SystemExit(f"no weights for seed {seed} — train first")
        entries = [(seed, w, False, source)]
    else:
        # config-declared seeds only — a lingering seed dir from another
        # experiment iteration must not silently join the aggregate
        cfg_seeds = cfg["meta"]["seeds"]
        entries = []
        for s in cfg_seeds:
            w, source = _evaluation_weights(exp_dir, cfg, s)
            if w:
                entries.append((s, w, False, source))
        extra = [s for s in _seeds_with_weights(exp_dir) if s not in cfg_seeds]
        if extra:
            print(f"ignoring seed dirs not declared in config.yaml: {extra} "
                  f"(use --seed to evaluate one explicitly)")
        if not entries:
            raise SystemExit(f"{exp_dir.name}: no trained config seeds found — train first")

    for s, w, custom, inherited_source in entries:
        print(f"== eval {exp_dir.name} seed {s} ({w})")
        if inherited_source:
            print(
                "   inherited checkpoint from "
                f"{inherited_source['experiment']} seed {inherited_source['seed']}"
            )
        tag = f"custom_{w.stem}" if custom else f"s{s}"
        dets_json = exp_dir / "artifacts" / f"predictions_{split}_{tag}.json"
        tiled_diagnostics = None
        evaluation_artifacts = {}
        if tiling:
            if custom_family:
                raise SystemExit("tiled inference is not implemented for ConvNeXt/Faster R-CNN")
            model = YOLO(str(w))
            from .sliced_inference import (full_frame_parity_suite, measure_tiled_speed,
                                           native_size_recall, predict_to_coco_tiled)
            print(
                "   tiled inference: "
                f"{tiling['tile_width']}x{tiling['tile_height']} source tiles, "
                f"overlap={tiling['overlap_width_ratio']:.2f}/"
                f"{tiling['overlap_height_ratio']:.2f}, global max_det={tiling['global_max_det']}"
            )
            manifest_json = exp_dir / "artifacts" / f"tile_manifest_{split}_{tag}.json"
            tiled_diag_json = exp_dir / "artifacts" / f"tiled_diagnostics_{split}_{tag}.json"
            parity_json = exp_dir / "artifacts" / f"control_parity_{split}_{tag}.json"
            parity = full_frame_parity_suite(model, list_images(dataset, split), proto, tiling)
            parity_json.write_text(json.dumps(parity, indent=2) + "\n")
            print(
                f"   full-frame parity PASS on {len(parity['cases'])} native shapes"
            )
            dets, tiled_diagnostics = predict_to_coco_tiled(
                model, dataset, split, proto, tiling, dets_json,
                manifest_json, tiled_diag_json,
            )
            tiled_diagnostics["native_size_recall"] = native_size_recall(gt_json, dets)
            tiled_diag_json.write_text(json.dumps(tiled_diagnostics, indent=2) + "\n")
            speed = measure_tiled_speed(model, dataset, split, proto, tiling)
            evaluation_artifacts.update({
                "control_parity": {"file": parity_json.name, "sha256": file_sha16(parity_json)},
                "tile_manifest": {"file": manifest_json.name, "sha256": file_sha16(manifest_json)},
                "tiled_diagnostics": {"file": tiled_diag_json.name, "sha256": file_sha16(tiled_diag_json)},
            })
        elif custom_family:
            from .convnext_frcnn import (measure_speed as measure_convnext_speed,
                                         predict_to_coco_checkpoint, validate_config)
            model, dets = predict_to_coco_checkpoint(
                cfg, w, dataset, split, dets_json
            )
            speed = measure_convnext_speed(
                model, dataset, split, proto, amp=validate_config(cfg)["amp"]
            )
            print(
                f"wrote {len(dets)} detections over {len(list_images(dataset, split))} "
                f"images -> {dets_json.name}"
            )
        else:
            model = YOLO(str(w))
            dets = predict_to_coco(model, dataset, split, proto, dets_json)
            speed = measure_speed(model, dataset, split, proto)
        evaluation_artifacts["predictions"] = {
            "file": dets_json.name,
            "sha256": file_sha16(dets_json),
        }
        evaluation_artifacts["experiment_config"] = {
            "file": "config.yaml",
            "sha256": file_sha16(exp_dir / "config.yaml"),
        }
        acc = coco_metrics(gt_json, dets, proto["max_det"], names)
        prediction_counts = Counter(d["image_id"] for d in dets)
        n_images = len(list_images(dataset, split))
        saturation = sum(n >= proto["max_det"] for n in prediction_counts.values())
        # model.info(verbose=False) returns None in this ultralytics version —
        # compute directly; None (not 0) on failure so gates flag it as missing
        if custom_family:
            from .convnext_frcnn import model_parameter_count
            n_p, flops = model_parameter_count(model), None
        else:
            try:
                from ultralytics.utils.torch_utils import get_flops, get_num_params
                n_p = get_num_params(model.model)
                flops = get_flops(model.model, proto["imgsz"])
            except Exception:
                n_p, flops = None, None
        protocol_record = {"hash": effective_hash,
                           "base_hash": protocol_hash(), "overrides": overrides,
                           "dataset": dataset, "split": split,
                           "gt": gt_fp, "imgsz": proto["imgsz"], "conf": proto["conf"],
                           "iou": proto["iou"], "max_det": proto["max_det"]}
        if identity_extra:
            protocol_record["inference"] = identity_extra
        diagnostics = {
            "images": n_images,
            "images_at_max_det": saturation,
            "pct_images_at_max_det": round(100 * saturation / n_images, 2),
        }
        if tiled_diagnostics:
            diagnostics["tiled"] = tiled_diagnostics
        model_record = {"params_m": round(n_p / 1e6, 3) if n_p else None,
                        "gflops": round(float(flops), 2) if flops else None,
                        "family": ((cfg.get("meta") or {}).get("model_family")
                                   or "ultralytics_yolo"),
                        "weights": str(w),
                        "weights_sha256": file_sha16(w),
                        "weights_source": inherited_source}
        if tiled_diagnostics and flops:
            model_record["gflops_per_forward"] = round(float(flops), 2)
            model_record["mean_forwards_per_image"] = tiled_diagnostics[
                "model_forwards_per_image_mean"
            ]
            model_record["estimated_pipeline_gflops_mean"] = round(
                float(flops) * tiled_diagnostics["model_forwards_per_image_mean"], 2
            )
        seed_metrics = {
            "protocol": protocol_record,
            "seed": None if custom else s,
            "custom_weights": str(w) if custom else None,
            "overall": acc["overall"],
            "per_class": acc["per_class"],
            "diagnostics": diagnostics,
            "artifacts": evaluation_artifacts,
            "speed": speed,
            "model": model_record,
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

    if not any(custom for _, _, custom, _ in entries):
        aggregate_exp(exp_dir)


def aggregate_exp(exp_dir: Path) -> None:
    """Merge per-seed metrics (same protocol hash) into experiment-level metrics.json."""
    cfg = load_config(exp_dir)
    expected_protocol_hash = effective_protocol_hash(cfg)
    allowed = {f"s{s}" for s in cfg["meta"]["seeds"]}
    per_seed, newest = {}, None
    skipped_cfg, skipped_stale = [], []
    for mj in sorted(exp_dir.glob("seeds/s*/metrics.json")):
        m = json.loads(mj.read_text())
        sk = f"s{m['seed']}"
        if m["protocol"]["hash"] != expected_protocol_hash:
            continue
        if sk not in allowed:
            skipped_cfg.append(sk)
            continue
        # metrics for a since-retrained checkpoint describe a dead model
        recorded = m["model"].get("weights_sha256")
        recorded_path = m.get("model", {}).get("weights")
        w = Path(recorded_path) if recorded_path else (
            exp_dir / "seeds" / sk / "weights" / "best.pt"
        )
        if recorded and w.exists() and file_sha16(w) != recorded:
            skipped_stale.append(sk)
            continue
        per_seed[sk] = m
        if newest is None or mj.stat().st_mtime > newest[0]:
            newest = (mj.stat().st_mtime, m["protocol"].get("gt"))
    if skipped_cfg:
        print(f"aggregation skipping seeds not in config: {skipped_cfg}")
    if skipped_stale:
        print(f"aggregation skipping retrained-since-eval seeds: {skipped_stale} — re-run eval")
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
        "diagnostics": first.get("diagnostics"),
        "artifacts": first.get("artifacts"),
        "speed": first["speed"],
        "model": first["model"],
        "seeds": {k: {"map50_95": v["overall"]["map50_95"],
                      "map50": v["overall"]["map50"],
                      "ap_small": v["overall"]["ap_small"],
                      "weights_sha256": v["model"].get("weights_sha256"),
                      "weights": v["model"].get("weights")}
                  for k, v in per_seed.items()},
        "env": first["env"],
    }
    (exp_dir / "metrics.json").write_text(json.dumps(exp_metrics, indent=2) + "\n")
    print(f"aggregated {len(per_seed)} seed(s) -> {exp_dir.name}/metrics.json")
