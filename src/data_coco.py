"""YOLO-format labels -> cached COCO ground-truth JSON.

One GT json per (dataset, split), cached under data/coco_gt/. Image ids are the
index into the sorted image filename list — stable across runs, shared by
COCOeval, TIDE, and the FiftyOne prediction attach. category_id = class idx + 1
(COCO convention: ids start at 1)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from .paths import COCO_GT_DIR, dataset_yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_dataset_cfg(dataset: str) -> dict:
    return yaml.safe_load(dataset_yaml(dataset).read_text())


def image_dir(dataset: str, split: str) -> Path:
    cfg = load_dataset_cfg(dataset)
    if split not in cfg or not cfg[split]:
        raise SystemExit(f"dataset '{dataset}' has no '{split}' split")
    return Path(cfg["path"]) / cfg[split]


def list_images(dataset: str, split: str) -> list[Path]:
    d = image_dir(dataset, split)
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images in {d} — run `make data` (visdrone) or train once (coco128 auto-download)")
    return imgs


def _label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def class_names(dataset: str) -> dict[int, str]:
    return {int(k): v for k, v in load_dataset_cfg(dataset)["names"].items()}


def build_gt(dataset: str, split: str, force: bool = False) -> Path:
    out = COCO_GT_DIR / f"{dataset}_{split}.json"
    if out.exists() and not force:
        return out
    COCO_GT_DIR.mkdir(parents=True, exist_ok=True)

    names = class_names(dataset)
    images, annotations = [], []
    ann_id = 1
    for img_id, p in enumerate(list_images(dataset, split)):
        with Image.open(p) as im:  # header read only
            w, h = im.size
        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lbl = _label_path(p)
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *(float(x) for x in f[1:5])
            ww, hh = bw * w, bh * h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": c + 1,
                "bbox": [round(cx * w - ww / 2, 2), round(cy * h - hh / 2, 2),
                         round(ww, 2), round(hh, 2)],
                "area": round(ww * hh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    out.write_text(json.dumps({
        "info": {"description": f"{dataset}/{split} GT (harness-generated)"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in sorted(names.items())],
    }))
    print(f"built {out.name}: {len(images)} images, {len(annotations)} boxes")
    return out
