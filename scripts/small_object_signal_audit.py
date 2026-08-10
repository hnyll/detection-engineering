#!/usr/bin/env python3
"""Build and score a GT-blinded small-object signal audit.

The build step selects a deterministic class/size-balanced sample of COCO-small
ground-truth boxes. For every object it renders the same source-coordinate field
of view from:

* a simulated full-image 640 letterbox,
* a simulated full-image 960 letterbox, and
* the native source image.

Triptych column order is randomized per object and stored only in answers.csv.
The static reviewer never receives the GT class or variant mapping. The score
step joins a completed review CSV with the answer key and reports descriptive
accuracy, coverage, paired rescues/losses, class breakdowns, and size-bin
breakdowns. This is a human information audit, not detector evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import paths  # noqa: E402
from src.data_coco import build_gt, gt_fingerprint, image_dir  # noqa: E402


CLASS_LABELS = ("person", "bicycle", "car", "truck", "bus", "motorcycle")
ABSTAIN_LABELS = ("uncertain", "not_visible")
ALLOWED_LABELS = set(CLASS_LABELS + ABSTAIN_LABELS)
VARIANTS = ("sim640", "sim960", "native")
VIEWS = ("a", "b", "c")
GRAY = (114, 114, 114)
MARKER = (255, 220, 0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def _load_experiment(ref: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    exp_dir = paths.resolve_exp(ref)
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("audit"), dict):
        raise SystemExit(f"{exp_dir.name}/config.yaml requires an audit section")
    if not (cfg.get("meta") or {}).get("analysis_only"):
        raise SystemExit(f"{exp_dir.name} must declare meta.analysis_only: true")
    return exp_dir, cfg, cfg["audit"]


def _output_dir(exp_dir: Path, raw: str | None) -> Path:
    if not raw:
        return exp_dir / "artifacts" / "signal_audit"
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _size_bin(short_side: float, bins: list[dict[str, Any]]) -> str | None:
    for spec in bins:
        lower = float(spec.get("min_px", 0))
        upper = spec.get("max_px")
        if short_side >= lower and (upper is None or short_side < float(upper)):
            return str(spec["name"])
    return None


def _balanced_targets(total: int, names: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(names))
    return {name: base + (1 if i < remainder else 0) for i, name in enumerate(names)}


def _select_samples(
    annotations: list[dict[str, Any]],
    categories: dict[int, str],
    bins: list[dict[str, Any]],
    samples_per_class: int,
    max_area: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if float(ann["area"]) >= max_area:
            continue
        cat_id = int(ann["category_id"])
        if categories.get(cat_id) not in CLASS_LABELS:
            continue
        w, h = (float(ann["bbox"][2]), float(ann["bbox"][3]))
        bin_name = _size_bin(min(w, h), bins)
        if bin_name is not None:
            grouped[(cat_id, bin_name)].append(ann)

    bin_names = [str(b["name"]) for b in bins]
    target_by_bin = _balanced_targets(samples_per_class, bin_names)
    jobs: list[tuple[int, int, str, int, list[dict[str, Any]]]] = []
    for cat_id, class_name in sorted(categories.items()):
        if class_name not in CLASS_LABELS:
            continue
        for bin_name in bin_names:
            candidates = list(grouped[(cat_id, bin_name)])
            rng.shuffle(candidates)
            target = target_by_bin[bin_name]
            if len(candidates) < target:
                raise SystemExit(
                    f"not enough {class_name}/{bin_name} candidates: "
                    f"need {target}, found {len(candidates)}"
                )
            jobs.append((len(candidates), cat_id, bin_name, target, candidates))

    # Select scarce strata first. This maximizes the chance of keeping one
    # audited object per source image across the full balanced sample.
    jobs.sort(key=lambda x: (x[0], categories[x[1]], x[2]))
    used_images: set[int] = set()
    selected_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    relaxations: list[dict[str, Any]] = []

    for _, cat_id, bin_name, target, candidates in jobs:
        chosen: list[dict[str, Any]] = []
        for ann in candidates:
            if int(ann["image_id"]) in used_images:
                continue
            chosen.append(ann)
            if len(chosen) == target:
                break

        if len(chosen) < target:
            need = target - len(chosen)
            for ann in candidates:
                if int(ann["id"]) in {int(x["id"]) for x in chosen}:
                    continue
                if int(ann["id"]) in selected_ids:
                    continue
                chosen.append(ann)
                if len(chosen) == target:
                    break
            relaxations.append({
                "class": categories[cat_id],
                "size_bin": bin_name,
                "reused_image_slots": need,
            })

        if len(chosen) != target:
            raise SystemExit(
                f"could not select {target} unique annotations for "
                f"{categories[cat_id]}/{bin_name}"
            )
        for ann in chosen:
            item = dict(ann)
            item["size_bin"] = bin_name
            selected.append(item)
            selected_ids.add(int(ann["id"]))
            used_images.add(int(ann["image_id"]))

    rng.shuffle(selected)
    return selected, relaxations


def _letterbox(image: Image.Image, size: int) -> tuple[Image.Image, float, int, int]:
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    dw, dh = size - new_w, size - new_h
    left = round(dw / 2 - 0.1)
    top = round(dh / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), GRAY)
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top


def _safe_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    width, height = max(1, right - left), max(1, bottom - top)
    out = Image.new("RGB", (width, height), GRAY)
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, right), min(image.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        patch = image.crop((src_left, src_top, src_right, src_bottom))
        out.paste(patch, (src_left - left, src_top - top))
    return out


def _draw_corner_marker(
    panel: Image.Image,
    bbox: tuple[float, float, float, float],
) -> None:
    draw = ImageDraw.Draw(panel)
    x1, y1, x2, y2 = bbox
    x1 = max(1, min(panel.width - 2, round(x1)))
    y1 = max(1, min(panel.height - 2, round(y1)))
    x2 = max(1, min(panel.width - 2, round(x2)))
    y2 = max(1, min(panel.height - 2, round(y2)))
    leg = max(6, min(14, round(min(abs(x2 - x1), abs(y2 - y1)) * 0.3)))
    width = 3
    segments = (
        ((x1, y1), (min(x2, x1 + leg), y1)),
        ((x1, y1), (x1, min(y2, y1 + leg))),
        ((x2, y1), (max(x1, x2 - leg), y1)),
        ((x2, y1), (x2, min(y2, y1 + leg))),
        ((x1, y2), (min(x2, x1 + leg), y2)),
        ((x1, y2), (x1, max(y1, y2 - leg))),
        ((x2, y2), (max(x1, x2 - leg), y2)),
        ((x2, y2), (x2, max(y1, y2 - leg))),
    )
    for p1, p2 in segments:
        draw.line((p1, p2), fill=MARKER, width=width)


def _render_panel(
    image: Image.Image,
    scale: float,
    pad_x: float,
    pad_y: float,
    center_x: float,
    center_y: float,
    context_side: float,
    native_bbox: tuple[float, float, float, float],
    display_size: int,
) -> Image.Image:
    cx = center_x * scale + pad_x
    cy = center_y * scale + pad_y
    side = max(1.0, context_side * scale)
    left = math.floor(cx - side / 2)
    top = math.floor(cy - side / 2)
    right = math.ceil(cx + side / 2)
    bottom = math.ceil(cy + side / 2)
    crop = _safe_crop(image, (left, top, right, bottom))
    panel = crop.resize((display_size, display_size), Image.Resampling.NEAREST)

    native_left = center_x - context_side / 2
    native_top = center_y - context_side / 2
    x, y, w, h = native_bbox
    marker = (
        (x - native_left) / context_side * display_size,
        (y - native_top) / context_side * display_size,
        (x + w - native_left) / context_side * display_size,
        (y + h - native_top) / context_side * display_size,
    )
    _draw_corner_marker(panel, marker)
    return panel


def _make_triptych(panels: list[Image.Image], sample_id: str, size: int) -> Image.Image:
    banner = 32
    out = Image.new("RGB", (size * 3, size + banner), (25, 25, 28))
    draw = ImageDraw.Draw(out)
    for i, panel in enumerate(panels):
        out.paste(panel, (i * size, banner))
        draw.text((i * size + 10, 9), f"VIEW {VIEWS[i].upper()}", fill=(245, 245, 245))
    draw.text((size * 3 - 62, 9), sample_id, fill=(170, 170, 175))
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _review_html(samples: list[dict[str, str]], manifest_key: str) -> str:
    samples_json = json.dumps(samples, separators=(",", ":"))
    labels_json = json.dumps(list(CLASS_LABELS + ABSTAIN_LABELS))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exp9 small-object signal audit</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1040px; margin: 24px auto; padding: 0 18px; background:#15161a; color:#eee; }}
  h1 {{ margin-bottom: 4px; }} .muted {{ color:#aaa; }}
  .bar {{ height:10px; background:#30323a; border-radius:9px; overflow:hidden; margin:14px 0; }}
  .bar > div {{ height:100%; background:#58a6ff; width:0; }}
  .card {{ background:#202228; border:1px solid #383b44; border-radius:12px; padding:16px; }}
  img {{ width:100%; image-rendering:pixelated; border-radius:6px; background:#727272; }}
  .labels {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:14px; }}
  label {{ display:flex; flex-direction:column; gap:6px; font-weight:650; }}
  select, textarea, button {{ font:inherit; color:#eee; background:#2a2d34; border:1px solid #4b4f5a; border-radius:6px; padding:9px; }}
  textarea {{ width:100%; min-height:58px; box-sizing:border-box; margin-top:12px; }}
  .buttons {{ display:flex; gap:8px; flex-wrap:wrap; margin:15px 0; }}
  button {{ cursor:pointer; }} button.primary {{ background:#1967b3; border-color:#388bfd; }}
  .warn {{ color:#f2cc60; }}
  @media (max-width:700px) {{ .labels {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>Exp9 small-object signal audit</h1>
<p class="muted">GT class and A/B/C resolution identities are hidden. The yellow corners mark the target. Choose independently for every view; use <b>uncertain</b> or <b>not_visible</b> rather than guessing.</p>
<p class="warn">Do not open <code>answers.csv</code> before exporting your review.</p>
<div id="progressText"></div><div class="bar"><div id="progressBar"></div></div>
<div class="card">
  <h2 id="sampleTitle"></h2>
  <img id="triptych" alt="three blinded views">
  <div class="labels" id="labels"></div>
  <textarea id="notes" placeholder="Optional notes: ambiguity, occlusion, context-only cue, questionable target..."></textarea>
</div>
<div class="buttons">
  <button id="prev">Previous</button><button id="next" class="primary">Save & next</button>
  <button id="unreviewed">Next unreviewed</button><button id="export">Export CSV</button>
  <button id="clear">Clear current</button>
</div>
<p class="muted">Progress is saved in this browser. Export periodically and at completion. Reviewing all 120 is preferred; the minimum planned analysis is 90 total with at least 15 GT examples per class, which can only be checked after scoring.</p>
<script>
const samples={samples_json};
const choices={labels_json};
const key='exp9-signal-audit-{manifest_key}';
let state=JSON.parse(localStorage.getItem(key)||'{{}}'); let idx=0;
const labels=document.getElementById('labels');
for (const v of ['a','b','c']) {{
  const lab=document.createElement('label'); lab.textContent='VIEW '+v.toUpperCase();
  const sel=document.createElement('select'); sel.id='label_'+v;
  sel.innerHTML='<option value="">Choose…</option>'+choices.map(x=>`<option value="${{x}}">${{x}}</option>`).join('');
  sel.addEventListener('change',save); lab.appendChild(sel); labels.appendChild(lab);
}}
document.getElementById('notes').addEventListener('input',save);
function reviewed(s) {{ const r=state[s.id]||{{}}; return ['a','b','c'].every(v=>r['label_'+v]); }}
function save() {{
  const s=samples[idx]; const r=state[s.id]||{{}};
  for (const v of ['a','b','c']) r['label_'+v]=document.getElementById('label_'+v).value;
  r.notes=document.getElementById('notes').value; state[s.id]=r;
  localStorage.setItem(key,JSON.stringify(state)); progress();
}}
function load() {{
  const s=samples[idx], r=state[s.id]||{{}};
  document.getElementById('sampleTitle').textContent=`${{idx+1}} / ${{samples.length}} — ${{s.id}}`;
  document.getElementById('triptych').src=s.triptych;
  for (const v of ['a','b','c']) document.getElementById('label_'+v).value=r['label_'+v]||'';
  document.getElementById('notes').value=r.notes||''; progress(); window.scrollTo(0,0);
}}
function progress() {{
  const n=samples.filter(reviewed).length;
  document.getElementById('progressText').textContent=`${{n}} / ${{samples.length}} triptychs complete`;
  document.getElementById('progressBar').style.width=(100*n/samples.length)+'%';
}}
document.getElementById('prev').onclick=()=>{{save();idx=(idx-1+samples.length)%samples.length;load();}};
document.getElementById('next').onclick=()=>{{save();idx=Math.min(samples.length-1,idx+1);load();}};
document.getElementById('unreviewed').onclick=()=>{{save();const j=samples.findIndex((s,i)=>i>idx&&!reviewed(s));const k=j>=0?j:samples.findIndex(s=>!reviewed(s));if(k>=0){{idx=k;load();}}}};
document.getElementById('clear').onclick=()=>{{state[samples[idx].id]={{}};localStorage.setItem(key,JSON.stringify(state));load();}};
function q(x) {{ return '"'+String(x??'').replaceAll('"','""')+'"'; }}
document.getElementById('export').onclick=()=>{{save();let rows=[['sample_id','triptych','label_a','label_b','label_c','notes']];for(const s of samples){{const r=state[s.id]||{{}};rows.push([s.id,s.triptych,r.label_a||'',r.label_b||'',r.label_c||'',r.notes||'']);}}const csv=rows.map(r=>r.map(q).join(',')).join('\\n')+'\\n';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='review_completed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}};
load();
</script>
</body></html>
"""


def build(args: argparse.Namespace) -> None:
    exp_dir, cfg, audit = _load_experiment(args.experiment)
    dataset = str(cfg["meta"]["dataset"])
    split = str(audit.get("split", "val"))
    seed = int(args.seed if args.seed is not None else audit.get("seed", 17))
    samples_per_class = int(
        args.samples_per_class
        if args.samples_per_class is not None
        else audit.get("samples_per_class", 20)
    )
    max_area = float(audit.get("coco_small_area_max", 1024))
    bins = list(audit["shortest_side_bins"])
    context_scale = float(audit.get("context_scale", 4.0))
    context_min = float(audit.get("minimum_context_px", 64))
    context_max = float(audit.get("maximum_context_px", 256))
    display_size = int(audit.get("display_size", 256))
    sim_sizes = [int(x) for x in audit.get("simulated_input_sizes", [640, 960])]
    if sim_sizes != [640, 960]:
        raise SystemExit("this audit currently requires simulated_input_sizes: [640, 960]")
    expected_gt = audit.get("expected_gt")
    actual_gt = gt_fingerprint(dataset, split)
    if expected_gt and str(expected_gt) != actual_gt:
        raise SystemExit(f"GT changed: expected {expected_gt}, got {actual_gt}")

    out = _output_dir(exp_dir, args.output)
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} already contains an audit — use --force to rebuild")
        shutil.rmtree(out)
    (out / "crops" / "triptych").mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        (out / "crops" / f"view_{view}").mkdir(parents=True, exist_ok=True)

    gt_path = build_gt(dataset, split)
    gt = json.loads(gt_path.read_text())
    categories = {int(c["id"]): str(c["name"]) for c in gt["categories"]}
    if tuple(categories[k] for k in sorted(categories)) != CLASS_LABELS:
        raise SystemExit(f"expected six classes {CLASS_LABELS}, got {categories}")
    images = {int(i["id"]): i for i in gt["images"]}
    source_dir = image_dir(dataset, split)
    rng = random.Random(seed)
    selected, relaxations = _select_samples(
        gt["annotations"], categories, bins, samples_per_class, max_area, rng
    )

    answer_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    html_samples: list[dict[str, str]] = []
    allocation: Counter[tuple[str, str]] = Counter()

    for index, ann in enumerate(selected, 1):
        sample_id = f"S{index:04d}"
        img_meta = images[int(ann["image_id"])]
        source = source_dir / str(img_meta["file_name"])
        with Image.open(source) as opened:
            native = opened.convert("RGB")
        x, y, w, h = (float(v) for v in ann["bbox"])
        cx, cy = x + w / 2, y + h / 2
        context_side = min(
            context_max,
            max(context_min, context_scale * max(w, h)),
        )
        prepared: dict[str, tuple[Image.Image, float, int, int]] = {
            "native": (native, 1.0, 0, 0),
        }
        for sim_size in sim_sizes:
            prepared[f"sim{sim_size}"] = _letterbox(native, sim_size)

        panels_by_variant: dict[str, Image.Image] = {}
        bbox = (x, y, w, h)
        for variant in VARIANTS:
            im, scale, pad_x, pad_y = prepared[variant]
            panels_by_variant[variant] = _render_panel(
                im, scale, pad_x, pad_y, cx, cy, context_side, bbox, display_size
            )

        variant_order = list(VARIANTS)
        rng.shuffle(variant_order)
        ordered_panels = [panels_by_variant[v] for v in variant_order]
        for view, panel in zip(VIEWS, ordered_panels):
            panel.save(out / "crops" / f"view_{view}" / f"{sample_id}.png")
        triptych_rel = f"crops/triptych/{sample_id}.png"
        _make_triptych(ordered_panels, sample_id, display_size).save(out / triptych_rel)

        class_name = categories[int(ann["category_id"])]
        allocation[(class_name, str(ann["size_bin"]))] += 1
        mapping = dict(zip(VIEWS, variant_order))
        answer_rows.append({
            "sample_id": sample_id,
            "image_id": int(ann["image_id"]),
            "file_name": img_meta["file_name"],
            "annotation_id": int(ann["id"]),
            "gt_class": class_name,
            "category_id": int(ann["category_id"]),
            "bbox_x": round(x, 3),
            "bbox_y": round(y, 3),
            "bbox_w": round(w, 3),
            "bbox_h": round(h, 3),
            "area": round(float(ann["area"]), 3),
            "native_short_side": round(min(w, h), 3),
            "size_bin": ann["size_bin"],
            "context_side": round(context_side, 3),
            "variant_a": mapping["a"],
            "variant_b": mapping["b"],
            "variant_c": mapping["c"],
        })
        review_rows.append({
            "sample_id": sample_id,
            "triptych": triptych_rel,
            "label_a": "",
            "label_b": "",
            "label_c": "",
            "notes": "",
        })
        html_samples.append({"id": sample_id, "triptych": triptych_rel})

    answer_fields = list(answer_rows[0])
    review_fields = ["sample_id", "triptych", "label_a", "label_b", "label_c", "notes"]
    answers_path = out / "answers.csv"
    template_path = out / "review_template.csv"
    _write_csv(answers_path, answer_fields, answer_rows)
    _write_csv(template_path, review_fields, review_rows)
    answers_sha = _sha256(answers_path)
    review_sha = _sha256(template_path)
    (out / "review.html").write_text(_review_html(html_samples, answers_sha[:16]))

    by_class_bin = {
        class_name: {bin_name: allocation[(class_name, bin_name)] for bin_name in [b["name"] for b in bins]}
        for class_name in CLASS_LABELS
    }
    report = {
        "experiment": exp_dir.name,
        "dataset": dataset,
        "split": split,
        "gt_fingerprint": actual_gt,
        "seed": seed,
        "parameters": {
            "samples_per_class": samples_per_class,
            "max_area_exclusive": max_area,
            "shortest_side_bins": bins,
            "context_scale": context_scale,
            "minimum_context_px": context_min,
            "maximum_context_px": context_max,
            "display_size": display_size,
            "simulated_input_sizes": sim_sizes,
        },
        "samples": len(answer_rows),
        "unique_source_images": len({int(r["image_id"]) for r in answer_rows}),
        "one_per_image_relaxations": relaxations,
        "allocation": by_class_bin,
        "answers_sha256": answers_sha,
        "review_template_sha256": review_sha,
        "review_blinding": {
            "gt_hidden": True,
            "variant_identity_hidden": True,
            "variant_column_order_randomized_per_sample": True,
            "independent_variant_blinding": False,
        },
    }
    _write_json(out / "build_report.json", report)
    print(f"built {len(answer_rows)} blinded triptychs from {report['unique_source_images']} images")
    print(f"review: {out / 'review.html'}")
    print(f"answer key (do not open yet): {answers_path}")
    if relaxations:
        print(f"one-image-per-object relaxation(s): {relaxations}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _metric(records: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = len(records)
    correct = sum(bool(r["correct"]) for r in records)
    covered = sum(bool(r["covered"]) for r in records)
    abstained = reviewed - covered
    return {
        "reviewed": reviewed,
        "correct": correct,
        "accuracy": round(correct / reviewed, 4) if reviewed else None,
        "accuracy_wilson_95": _wilson(correct, reviewed),
        "coverage": round(covered / reviewed, 4) if reviewed else None,
        "accuracy_when_answered": round(correct / covered, 4) if covered else None,
        "uncertain_or_not_visible": abstained,
        "uncertain_rate": round(abstained / reviewed, 4) if reviewed else None,
    }


def _macro_accuracy(records: list[dict[str, Any]]) -> float | None:
    vals = []
    for class_name in CLASS_LABELS:
        subset = [r for r in records if r["gt_class"] == class_name]
        if subset:
            vals.append(sum(bool(r["correct"]) for r in subset) / len(subset))
    return round(sum(vals) / len(vals), 4) if vals else None


def _paired(
    by_sample_variant: dict[tuple[str, str], dict[str, Any]],
    higher: str,
    lower: str,
) -> dict[str, Any]:
    sample_ids = sorted({sample for sample, _ in by_sample_variant})
    pairs = [
        (by_sample_variant[(s, higher)], by_sample_variant[(s, lower)])
        for s in sample_ids
        if (s, higher) in by_sample_variant and (s, lower) in by_sample_variant
    ]
    if not pairs:
        return {"n": 0, "accuracy_delta": None, "rescued": 0, "lost": 0, "rescue_loss_ratio": None}
    high_correct = sum(bool(a["correct"]) for a, _ in pairs)
    low_correct = sum(bool(b["correct"]) for _, b in pairs)
    rescued = sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in pairs)
    lost = sum(not bool(a["correct"]) and bool(b["correct"]) for a, b in pairs)
    ratio = None if lost == 0 else round(rescued / lost, 3)
    if lost == 0 and rescued:
        ratio = "inf"
    return {
        "n": len(pairs),
        "accuracy_delta": round((high_correct - low_correct) / len(pairs), 4),
        "rescued": rescued,
        "lost": lost,
        "rescue_loss_ratio": ratio,
    }


def score(args: argparse.Namespace) -> None:
    exp_dir, _, _ = _load_experiment(args.experiment)
    out = _output_dir(exp_dir, args.output)
    answers_path = out / "answers.csv"
    build_report_path = out / "build_report.json"
    review_path = Path(args.review) if args.review else out / "review_completed.csv"
    if not review_path.is_absolute():
        review_path = ROOT / review_path
    for required in (answers_path, build_report_path, review_path):
        if not required.exists():
            raise SystemExit(f"missing {required}")

    answers = {r["sample_id"]: r for r in _read_csv(answers_path)}
    reviews_raw = _read_csv(review_path)
    reviews: dict[str, dict[str, str]] = {}
    for row in reviews_raw:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in reviews:
            raise SystemExit(f"duplicate review row for {sample_id}")
        if sample_id not in answers:
            raise SystemExit(f"unknown sample_id in review: {sample_id}")
        reviews[sample_id] = row

    records: list[dict[str, Any]] = []
    invalid: list[str] = []
    completed_samples: Counter[str] = Counter()
    for sample_id, row in reviews.items():
        answer = answers[sample_id]
        all_present = True
        for view in VIEWS:
            label = (row.get(f"label_{view}") or "").strip().lower()
            if not label:
                all_present = False
                continue
            if label not in ALLOWED_LABELS:
                invalid.append(f"{sample_id}/view_{view}={label!r}")
                continue
            variant = answer[f"variant_{view}"]
            records.append({
                "sample_id": sample_id,
                "variant": variant,
                "view": view,
                "label": label,
                "gt_class": answer["gt_class"],
                "size_bin": answer["size_bin"],
                "correct": label == answer["gt_class"],
                "covered": label in CLASS_LABELS,
            })
        if all_present:
            completed_samples[answer["gt_class"]] += 1
    if invalid:
        raise SystemExit("invalid labels:\n  " + "\n  ".join(invalid[:20]))
    if not records:
        raise SystemExit("review contains no completed labels")

    overall: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [r for r in records if r["variant"] == variant]
        overall[variant] = _metric(subset)
        overall[variant]["macro_accuracy"] = _macro_accuracy(subset)

    per_class: dict[str, Any] = {}
    for class_name in CLASS_LABELS:
        per_class[class_name] = {
            variant: _metric([
                r for r in records
                if r["variant"] == variant and r["gt_class"] == class_name
            ])
            for variant in VARIANTS
        }
    build_report = json.loads(build_report_path.read_text())
    bin_names = [b["name"] for b in build_report["parameters"]["shortest_side_bins"]]
    per_size_bin: dict[str, Any] = {}
    for bin_name in bin_names:
        per_size_bin[bin_name] = {
            variant: _metric([
                r for r in records
                if r["variant"] == variant and r["size_bin"] == bin_name
            ])
            for variant in VARIANTS
        }

    by_sample_variant = {(r["sample_id"], r["variant"]): r for r in records}
    paired = {
        "sim960_minus_sim640": _paired(by_sample_variant, "sim960", "sim640"),
        "native_minus_sim960": _paired(by_sample_variant, "native", "sim960"),
        "native_minus_sim640": _paired(by_sample_variant, "native", "sim640"),
    }
    min_per_class = min(completed_samples.get(c, 0) for c in CLASS_LABELS)
    native_macro = overall["native"]["macro_accuracy"]
    native_coverage = overall["native"]["coverage"]
    native_640_delta = paired["native_minus_sim640"]["accuracy_delta"]
    sim_delta = paired["sim960_minus_sim640"]["accuracy_delta"]
    criteria = {
        "reviewed_at_least_15_per_class": {
            "observed_min": min_per_class,
            "threshold": 15,
            "pass": min_per_class >= 15,
        },
        "native_macro_accuracy": {
            "observed": native_macro,
            "threshold": 0.50,
            "pass": native_macro is not None and native_macro >= 0.50,
        },
        "native_coverage": {
            "observed": native_coverage,
            "threshold": 0.70,
            "pass": native_coverage is not None and native_coverage >= 0.70,
        },
        "native_minus_sim640_accuracy": {
            "observed": native_640_delta,
            "threshold": 0.15,
            "pass": native_640_delta is not None and native_640_delta >= 0.15,
        },
        "sim960_minus_sim640_accuracy": {
            "observed": sim_delta,
            "threshold": 0.08,
            "pass": sim_delta is not None and sim_delta >= 0.08,
        },
    }
    report = {
        "experiment": exp_dir.name,
        "review_file": str(review_path),
        "review_sha256": _sha256(review_path),
        "answer_key_sha256": _sha256(answers_path),
        "completed_triptychs": sum(1 for c in reviews if all((reviews[c].get(f"label_{v}") or "").strip() for v in VIEWS)),
        "completed_triptychs_by_gt_class": {c: completed_samples.get(c, 0) for c in CLASS_LABELS},
        "overall": overall,
        "paired": paired,
        "per_class": per_class,
        "per_size_bin": per_size_bin,
        "criteria": criteria,
        "limitations": [
            "balanced audit sample is not prevalence-weighted",
            "one reviewer is blinded to GT and variant identities but sees all three variants simultaneously",
            "target location is cued, so search/localization/NMS are not tested",
            "human recognizability is evidence of information availability, not detector learnability",
        ],
    }
    _write_json(out / "score_report.json", report)

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    md = [
        "# Exp9 small-object signal audit score", "",
        f"Reviewed complete triptychs: **{report['completed_triptychs']} / {build_report['samples']}**", "",
        "## Overall", "",
        "| Variant | Reviewed | Accuracy | Macro accuracy | Coverage | Accuracy when answered | Uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display_names = {"sim640": "simulated 640", "sim960": "simulated 960", "native": "native"}
    for variant in VARIANTS:
        m = overall[variant]
        md.append(
            f"| {display_names[variant]} | {m['reviewed']} | {pct(m['accuracy'])} | "
            f"{pct(m['macro_accuracy'])} | {pct(m['coverage'])} | "
            f"{pct(m['accuracy_when_answered'])} | {m['uncertain_or_not_visible']} |"
        )
    md += ["", "## Paired effects", "", "| Comparison | N | Accuracy delta | Rescued | Lost | Rescue:loss |", "|---|---:|---:|---:|---:|---:|"]
    for name, p in paired.items():
        md.append(f"| {name} | {p['n']} | {pct(p['accuracy_delta'])} | {p['rescued']} | {p['lost']} | {p['rescue_loss_ratio']} |")
    md += ["", "## Preregistered checks", "", "| Criterion | Observed | Required | Result |", "|---|---:|---:|---|"]
    for name, c in criteria.items():
        observed = c.get("observed", c.get("observed_min"))
        obs_s = str(observed) if "reviewed" in name else pct(observed)
        req_s = str(c["threshold"]) if "reviewed" in name else pct(c["threshold"])
        md.append(f"| {name} | {obs_s} | {req_s} | {'PASS' if c['pass'] else 'FAIL'} |")
    md += ["", "## Per-class accuracy", "", "| Class | 640 | 960 | Native | Complete triptychs |", "|---|---:|---:|---:|---:|"]
    for class_name in CLASS_LABELS:
        md.append(
            f"| {class_name} | {pct(per_class[class_name]['sim640']['accuracy'])} | "
            f"{pct(per_class[class_name]['sim960']['accuracy'])} | "
            f"{pct(per_class[class_name]['native']['accuracy'])} | "
            f"{completed_samples.get(class_name, 0)} |"
        )
    md += ["", "## By native shortest-side bin", "", "| Bin | 640 | 960 | Native |", "|---|---:|---:|---:|"]
    for bin_name in bin_names:
        md.append(
            f"| {bin_name} | {pct(per_size_bin[bin_name]['sim640']['accuracy'])} | "
            f"{pct(per_size_bin[bin_name]['sim960']['accuracy'])} | "
            f"{pct(per_size_bin[bin_name]['native']['accuracy'])} |"
        )
    md += ["", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    (out / "score_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:24]))
    print(f"\nwrote {out / 'score_report.json'}")
    print(f"wrote {out / 'score_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build", help="sample and render a blinded crop audit")
    build_p.add_argument("--experiment", required=True)
    build_p.add_argument("--output", default=None)
    build_p.add_argument("--seed", type=int, default=None)
    build_p.add_argument("--samples-per-class", type=int, default=None)
    build_p.add_argument("--force", action="store_true")
    score_p = sub.add_parser("score", help="join a completed review CSV with the hidden key")
    score_p.add_argument("--experiment", required=True)
    score_p.add_argument("--review", default=None)
    score_p.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
