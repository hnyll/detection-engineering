#!/usr/bin/env python3
"""Build a YOLO dataset view with remapped class ids.

Images are hardlinked when possible and otherwise symlinked (with copying as a
last resort); only label files are generated. This keeps the original dataset
untouched while making a class-grouping CSV the single source of truth for a
taxonomy experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_mapping(path: Path, source_names: dict[int, str]) -> tuple[dict[int, int], dict[int, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"source_class", "group_id", "group_name"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"{path}: CSV must contain {sorted(required)} columns")

    by_name: dict[str, tuple[int, str]] = {}
    for row in rows:
        name = row["source_class"].strip()
        if not name:
            raise SystemExit(f"{path}: source_class cannot be blank")
        try:
            gid = int(row["group_id"])
        except ValueError as e:
            raise SystemExit(f"{path}: invalid group_id for {name!r}") from e
        gname = row["group_name"].strip()
        if gid < 0 or not gname:
            raise SystemExit(f"{path}: invalid group for {name!r}")
        if name in by_name:
            raise SystemExit(f"{path}: duplicate source_class {name!r}")
        by_name[name] = (gid, gname)

    expected = set(source_names.values())
    missing = expected - set(by_name)
    extra = set(by_name) - expected
    if missing or extra:
        raise SystemExit(f"{path}: missing={sorted(missing)} extra={sorted(extra)}")

    group_ids = {gid for gid, _ in by_name.values()}
    if group_ids != set(range(max(group_ids) + 1)):
        raise SystemExit(f"{path}: group_id values must be contiguous from 0 (got {sorted(group_ids)})")

    name_to_id = {name: idx for idx, name in source_names.items()}
    remap = {name_to_id[name]: gid for name, (gid, _) in by_name.items()}
    group_names = {gid: name for gid, name in by_name.values()}
    if len(group_names) != len(group_ids):
        raise SystemExit(f"{path}: each group_id must have exactly one group_name")
    return remap, group_names


def materialize_images(src: Path, dst: Path) -> None:
    """Expose images under the derived path without duplicating their bytes.

    Directory symlinks are not sufficient here: Ultralytics resolves an image's
    real path before deriving its label path, which would send it back to the
    source (10-class) labels. Hardlinks preserve the bytes while keeping the
    derived image path paired with the remapped labels. Per-file symlinks are
    used across filesystems, with copying only as a final fallback.
    """
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for image in sorted(src.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        target = dst / image.name
        try:
            os.link(image, target)
        except OSError:
            try:
                target.symlink_to(image.resolve())
            except OSError:
                shutil.copy2(image, target)


def build(source_yaml: Path, mapping_csv: Path, output: Path) -> None:
    source = yaml.safe_load(source_yaml.read_text())
    source_names = {int(k): str(v) for k, v in source["names"].items()}
    remap, group_names = load_mapping(mapping_csv, source_names)
    source_root = Path(source["path"]).expanduser().resolve()
    if not (source_root / "images" / "train").is_dir():
        raise SystemExit(f"source dataset is missing images/train: {source_root}")

    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src_images = source_root / "images" / split
        src_labels = source_root / "labels" / split
        if not src_images.exists():
            continue
        materialize_images(src_images, output / "images" / split)
        out_labels = output / "labels" / split
        if out_labels.exists():
            shutil.rmtree(out_labels)
        out_labels.mkdir(parents=True, exist_ok=True)
        for src_label in sorted(src_labels.glob("*.txt")):
            lines = []
            for raw in src_label.read_text().splitlines():
                if not raw.strip():
                    continue
                fields = raw.split()
                if len(fields) < 5:
                    raise SystemExit(f"malformed label line in {src_label}: {raw!r}")
                old = int(fields[0])
                if old not in remap:
                    raise SystemExit(f"class id {old} missing from {mapping_csv}")
                fields[0] = str(remap[old])
                lines.append(" ".join(fields))
            (out_labels / src_label.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    generated = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {gid: name for gid, name in sorted(group_names.items())},
        "source_dataset_yaml": str(source_yaml.resolve()),
        "mapping_csv": str(mapping_csv.resolve()),
        "mapping_sha256": sha16(mapping_csv),
    }
    out_yaml = output / "dataset.yaml"
    out_yaml.write_text(yaml.safe_dump(generated, sort_keys=False))
    print(f"built {output} using {mapping_csv} ({len(group_names)} groups)")
    print(f"dataset config: {out_yaml}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-yaml", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.source_yaml, args.mapping, args.output)


if __name__ == "__main__":
    main()
