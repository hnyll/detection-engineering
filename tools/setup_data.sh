#!/usr/bin/env bash
# Idempotent: links the existing local VisDrone copy (YOLO format) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=/home/hnyll/objdet_exp/data/visdrone/raw
mkdir -p data

if [ -e data/visdrone ]; then
    echo "data/visdrone already exists -> $(readlink -f data/visdrone)"
else
    [ -d "$SRC/images/val" ] || { echo "ERROR: VisDrone source not found at $SRC" >&2; exit 1; }
    ln -s "$SRC" data/visdrone
    echo "linked data/visdrone -> $SRC"
fi

echo "val images: $(ls data/visdrone/images/val/*.jpg 2>/dev/null | wc -l) (expect 548)"
