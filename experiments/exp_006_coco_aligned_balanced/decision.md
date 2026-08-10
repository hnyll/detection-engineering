---
exp: "exp_006_coco_aligned_balanced"
hypothesis_confirmed: true
outcome: positive
next_experiment: "exp_007_inference_960"
---

# exp_006_coco_aligned_balanced — Decision

## Problem

The official ten-class taxonomy contains visually ambiguous boundaries, while
rare and tiny classes receive limited useful visual evidence. Exp6 tested a
coarser six-class COCO-like application taxonomy together with the class-aware
sampling policy from Exp5.

## Result

- Fixed-protocol mAP50-95 is **0.21767**, mAP50 **0.37672**, and mAP75
  **0.21391**.
- AP-small is **0.11139**, versus **0.32413** medium and **0.51011** large.
- TIDE's main errors are classification **14.879 dAP**, misses **9.288**, and
  localization **6.252**. Special FalseNeg impact is **30.890**.
- Car is strongest at **0.51765 AP50-95**. Bicycle remains near the floor at
  **0.02930**; person, truck, and motorcycle are around **0.15-0.16**.
- Descriptively, Exp6 is +0.04502 mAP50-95 and -8.371 Cls dAP relative to Exp5.
  Because the ground-truth taxonomy changed, these are application-task
  indicators rather than an official VisDrone comparison.

The combined custom-task hypothesis is confirmed for seed 17: the grouped
person, car, and motorcycle concepts remain usable, aggregate AP increased
substantially, and classification impact decreased. This is provisional rather
than statistical evidence because Exp6 has one seed and combines taxonomy with
recomputed sampling factors.

## What the confusion matrix says

Cars, trucks, and buses are separate Exp6 classes; only car and van were merged.
At the matrix's confidence-0.25/IoU-0.45 operating point, truck to car is the
largest visible semantic confusion. The much larger bottom background row is
unmatched ground truth, not a background class predicted by YOLO. It indicates
low-confidence, missed, or insufficiently overlapping boxes and is especially
large for bicycle, person, and motorcycle.

The source-detail hypothesis is measurable. At imgsz 640, median input boxes
are only 5.6 x 12.3 px for person, 9.4 x 10.4 for bicycle, and 10.4 x 11.3 for
motorcycle. A 10 x 10 box shifted by two pixels in both axes has IoU about 0.47,
so the same resolution bottleneck can create both misses and localization
errors.

## Possible causes (ranked)

1. **Too few effective pixels.** Full-image resizing compresses many targets
   below a size where appearance and exact boundaries are reliable.
2. **Remaining class ambiguity.** Truck/car and bus/car/truck distinctions are
   still required by the COCO-like taxonomy and remain difficult from aerial
   viewpoints.
3. **Broad rather than exact balancing.** Image repetition also repeats cars
   and people that co-occur with the target class; it adds exposure but not new
   visual detail.
4. **Approximate mapping.** Three-wheel vehicles are assigned to motorcycle
   because COCO provides no direct target, which may increase within-class
   variation.
5. **Annotation precision.** Tiny or occluded boxes may have an unavoidable IoU
   floor, but no Exp6 audit quantified this.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Inference resolution | Evaluate the same checkpoint at imgsz 960 | Select for Exp7; isolates retained input detail without retraining |
| Tiled inference | Overlapping source-image tiles | Next candidate if 960 helps; separate experiment because it changes passes and merging |
| Crop/tile training | Object-centered crops with surrounding context | Use only after the inference diagnostic supports the mechanism |
| Model capacity | Pretrained YOLO11s at 640 | Valid later semantic-capacity ablation; more expensive and does not add source pixels |
| Data filtering | Remove tiny labels | Reject unless the application consistently ignores the same objects in train and evaluation |
| Loss | Increase classification or box-loss weight | Defer; loss weights do not create missing visual information |
| NMS/assignment | Tune NMS or replace assignment | Defer; duplicate impact is small and assignment failure is not measured |
| Taxonomy | Merge car/truck/bus into road_vehicle | Optional application choice, but no longer the intended COCO-like six-class task |

## Tradeoffs

Inference at 960 processes 2.25x as many square input pixels as 640. It should
retain more detail but increases latency, VRAM, and GFLOPs. It is an inference
protocol change, so its result cannot replace the fixed-640 benchmark silently.
Tiling can magnify objects further but requires multiple passes, overlap, and
duplicate merging.

The Exp6 prediction file already reaches `max_det=300` on 458 of 548 validation
images at the 0.001 confidence floor. Exp7 must keep that cap fixed to isolate
resolution, but cap saturation should be reported if recall gains appear
limited.

## Exp7 validation plan

Exp7 reuses the exact Exp6 seed-17 checkpoint (`eaed04c240d52e73`) and six-class
ground truth (`e1142868d4dc2e9d`). Only prediction `imgsz` changes from 640 to
960; confidence 0.001, NMS IoU 0.7, max detections 300, batch 1, `rect=False`,
and the evaluation code remain fixed. No training, tiling, or TTA is included.

Primary success criterion:

- AP-small improves from 0.11139 to at least **0.12139**.

Supporting criteria:

- AR-max improves from 0.29215 to at least **0.30215**.
- TIDE Miss decreases from 9.288 to at most **8.288**.
- TIDE Loc decreases from 6.252 to at most **5.752**.

The primary criterion plus at least two supporting criteria must pass. Guardrails:

- mAP50-95 must remain at least **0.21267**.
- Cls dAP must not exceed **15.879**.
- special FalsePos dAP must not exceed **14.493**.
- mean batch-1 latency must not exceed **25.15 ms** for the provisional
  deployment budget.

## Recommendation

Adopt Exp6 as the current custom six-class application baseline and retain Exp5
separately for official ten-class reporting. Proceed with the inference-only
Exp7 resolution diagnostic. Do not delete tiny annotations or tune loss/NMS
before testing whether the same checkpoint benefits from more retained input
detail.
