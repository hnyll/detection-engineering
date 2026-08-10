#!/usr/bin/env python3
"""Build a deterministic repeat-factor sampling manifest for YOLO training.

The manifest repeats paths; it does not copy images or labels. Repeat factors
follow the LVIS/Detectron2-style rule r(c) = max(1, sqrt(t / f(c))), where f(c)
is the fraction of source images containing class c. An image receives the
largest factor among its classes, capped by --max-repeat. Fractional factors are
rounded deterministically from --seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path has no 'images' component: {image}")


def classes_and_counts(image: Path, nc: int) -> tuple[set[int], Counter[int]]:
    classes: set[int] = set()
    counts: Counter[int] = Counter()
    label = label_path(image)
    if not label.exists():
        return classes, counts
    for line_no, line in enumerate(label.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            cls = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{label}:{line_no}: invalid class id") from exc
        if not 0 <= cls < nc:
            raise ValueError(f"{label}:{line_no}: class {cls} outside [0, {nc})")
        classes.add(cls)
        counts[cls] += 1
    return classes, counts


def deterministic_unit_interval(seed: int, name: str) -> float:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yaml", type=Path, default=Path("configs/visdrone.yaml"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--repeat-threshold", type=float, default=0.20)
    parser.add_argument("--max-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 0 < args.repeat_threshold <= 1:
        raise SystemExit("--repeat-threshold must be in (0, 1]")
    if args.max_repeat < 1:
        raise SystemExit("--max-repeat must be >= 1")

    source = yaml.safe_load(args.source_yaml.read_text())
    names = {int(k): str(v) for k, v in source["names"].items()}
    nc = len(names)
    source_root = Path(source["path"]).expanduser().resolve()
    image_dir = source_root / source[args.split]
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"no images found in {image_dir}")
    records = []
    image_counts: Counter[int] = Counter()
    instance_counts: Counter[int] = Counter()
    for image in images:
        classes, counts = classes_and_counts(image, nc)
        records.append((image, classes, counts))
        image_counts.update(classes)
        instance_counts.update(counts)

    frequencies = {c: image_counts[c] / len(images) for c in range(nc)}
    class_factors = {
        c: min(
            float(args.max_repeat),
            max(1.0, math.sqrt(args.repeat_threshold / frequencies[c]))
            if frequencies[c] > 0
            else float(args.max_repeat),
        )
        for c in range(nc)
    }

    manifest: list[str] = []
    copy_histogram: Counter[int] = Counter()
    sampled_image_counts: Counter[int] = Counter()
    sampled_instance_counts: Counter[int] = Counter()
    for image, classes, counts in records:
        factor = max((class_factors[c] for c in classes), default=1.0)
        whole = math.floor(factor)
        copies = whole + (
            deterministic_unit_interval(args.seed, image.name) < factor - whole
        )
        copies = max(1, min(args.max_repeat, copies))
        # Keep the dataset-view path instead of resolving image symlinks. The
        # view's sibling labels/ directory contains the remapped annotations;
        # resolving here would incorrectly pair images with source labels.
        manifest.extend([str(image.absolute())] * copies)
        copy_histogram[copies] += 1
        for cls in classes:
            sampled_image_counts[cls] += copies
        for cls, count in counts.items():
            sampled_instance_counts[cls] += count * copies

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(manifest) + "\n")
    manifest_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]
    report_path = args.report or args.output.with_suffix(".report.json")
    report = {
        "method": "repeat_factor_sampling",
        "formula": "r(c)=max(1,sqrt(repeat_threshold/image_frequency(c)))",
        "source_yaml": str(args.source_yaml),
        "dataset": args.source_yaml.stem,
        "split": args.split,
        "seed": args.seed,
        "repeat_threshold": args.repeat_threshold,
        "max_repeat": args.max_repeat,
        "source_images": len(images),
        "sampled_entries": len(manifest),
        "expansion_ratio": round(len(manifest) / len(images), 5),
        "copy_histogram": {str(k): copy_histogram[k] for k in sorted(copy_histogram)},
        "manifest": str(args.output),
        "manifest_sha256": manifest_sha,
        "classes": {
            names[c]: {
                "id": c,
                "source_images": image_counts[c],
                "source_image_frequency": round(frequencies[c], 6),
                "source_instances": instance_counts[c],
                "class_repeat_factor": round(class_factors[c], 5),
                "sampled_image_entries": sampled_image_counts[c],
                "sampled_instances": sampled_instance_counts[c],
            }
            for c in range(nc)
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"wrote {len(manifest)} entries from {len(images)} images "
        f"({report['expansion_ratio']:.2f}x) -> {args.output}"
    )
    print("class                 src imgs  src boxes  factor  sampled boxes")
    for c in range(nc):
        print(
            f"{names[c]:<21} {image_counts[c]:>8} {instance_counts[c]:>10} "
            f"{class_factors[c]:>7.2f} {sampled_instance_counts[c]:>14}"
        )
    print(f"report -> {report_path} (manifest {manifest_sha})")


if __name__ == "__main__":
    main()
