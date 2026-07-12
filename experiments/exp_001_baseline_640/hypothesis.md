# exp_001_baseline_640 — Hypothesis

## Question
What does a stock YOLO11n at 640px, trained 100 epochs on VisDrone-DET train,
achieve on the fixed-protocol val split — and which failure modes dominate?

## Prediction (write BEFORE training)
Written 2026-07-12, before any training run of this experiment:

- **mAP50-95 ≈ 0.18–0.22**, mAP50 ≈ 0.32–0.38 (spec expects "roughly 20 mAP"
  from a stock baseline; SOTA ≈ 40).
- **AP_small < 0.10** and far below AP_large — most VisDrone objects are <32px,
  below what a stride-8 P3 head resolves well at 640px input.
- Per-class: **car strongest** (largest, most frequent class), pedestrian/people
  weak, **bicycle/tricycle/awning-tricycle weakest** (rare + small + confusable).
- TIDE: **Miss and Bkg dominate** (tiny objects not assigned/detected; dense
  scenes produce background FPs), with substantial Loc error; Cls confusion
  concentrated in {pedestrian,people} and the tricycle family.
- Seed variance on mAP50-95 ≈ ±0.002–0.005 — improvements smaller than this in
  later ablations are noise.
- Latency ≈ 10–25 ms/img batch=1 on the 4060 (phase0 measured 9–14 ms on
  COCO128; VisDrone's denser NMS load will add time).

## Background reading / prior evidence
- docs/harness_spec.md Phase 1: 6,471 train / 548 val images, 10 classes, dense
  tiny objects, heavy class imbalance; low baseline numbers are the point.
- Small-object priors: objects <32px occupy <4px on the P3 feature map at
  640px; resolution (imgsz) and tiling (SAHI) are the planned Phase 1 ablations.
- phase0 (exp_000) verified the plumbing only — no modeling conclusions carried.

## Baseline
None — this experiment IS the Phase 1 baseline (parent_exp: null).

## Causal variable changed
None — baseline establishment with stock settings.

## Compensating implementation changes
None. Machine-safe defaults apply (batch 16 @ 640, workers 2, AMP), which are
the standard settings this baseline defines for all descendants.
