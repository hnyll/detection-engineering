---
exp: "exp_011_tiled_inference_640"
hypothesis_confirmed: false
outcome: positive
next_experiment: "exp_012_hybrid_fullframe_tiles_640"
---

# exp_011_tiled_inference_640 — Decision

## Problem

Exp10 left 59.28% of native `<8 px` targets unmatched. Exp11 tested whether
their effective feature scale, rather than class weighting alone, was limiting
the detector. It reused the exact Exp10 seed-17 checkpoint and replaced the
full-frame pass with overlapping native-source tiles. No training changed.

## Result

The scale mechanism is strongly supported. AP-small increased from 0.17903 to
0.23582, native `<8 px` recall from 0.40717 to 0.63973, AR from 0.38521 to
0.45216, and mAP50-95 from 0.29311 to 0.31128. TIDE Miss improved by 1.546 dAP
and Cls by 1.801 dAP. The latter is important: some apparent classification
weakness was coupled to feature scale or clutter, not just class frequency.

The exact pipeline also has serious costs. AP-medium fell by 0.01913 and
AP-large by 0.17642. TIDE Dupe worsened by 1.000 dAP and special FalsePos by
5.509 dAP. The final top-300 budget was reached on 97.81% of images at the AP
confidence floor, and 69.85% of individual tiles reached their local cap in
the mostly very-low-score tail. Mean end-to-end latency was 66.40 ms, or 15.1
FPS, and still passed the preregistered diagnostic latency gate.

## Preregistered decision

Both co-primary targets passed and three of four supporting targets passed.
However, the rule required all guardrails to pass, and AP-medium, AP-large,
Dupe, FalsePos, and max-det saturation failed. `hypothesis_confirmed` is
therefore `false` under the preregistered composite rule.

The practical outcome is `positive` as a mechanism experiment: the effect on
tiny-object recall and AP is too large and consistent to dismiss. Positive does
not mean unconditional adoption. Keep Exp10 full-frame as the general-purpose
control and treat Exp11 as a recall-oriented candidate until context, false
positives, and merge pressure are addressed.

## What the experiment teaches

1. **Tiny feature scale is causal for this checkpoint.** The frozen model
   recovers substantially more `<8 px` and `8-15 px` targets when source crops
   are magnified.
2. **Classification is partly upstream of visibility.** Tiling reduced Cls dAP
   far more than inverse-frequency class weighting did, so further class-loss
   tuning is not the best immediate move.
3. **Pure crops lose context and fragment large objects.** This is the leading
   explanation for the AP-large regression, though Exp11 bundles crop context,
   boundaries, and merging and cannot isolate them individually.
4. **Candidate management is now a bottleneck.** Overlapping views increase
   duplicates, false positives, and global-cap competition. NMS should not be
   blindly tuned on the same validation set; the next experiment should change
   one pipeline component and retain the fixed evaluation floor.

## Operating threshold

On Exp11's frozen predictions, confidence 0.40 is the best tested global F1
point: micro precision 0.7325, recall 0.6419, F1 0.6842, and 16.58 FP/image.
At 0.25, recall is 0.7322 but FP/image is 37.80. This threshold sweep is useful
for choosing a deployment operating point, but it does not alter COCO AP or
TIDE and was selected on the validation set. Confirm any chosen threshold on
untouched labeled data.

## Tradeoffs

| Choice | Benefit | Cost |
|---|---|---|
| Exp10 full-frame 960 | strong large-object/context behavior; 94.7 FPS | weaker tiny recall and AP-small |
| Exp11 pure 640 tiles | substantially better tiny recall, AP-small, AR, and overall mAP | 15.1 FPS, large-object regression, more FP/duplicates/cap pressure |
| lower confidence threshold | higher operating recall | rapidly increasing FP/image; no model improvement |
| higher confidence threshold | fewer false alarms | lower recall, especially for bicycle and other tiny classes |

## Next experiment

Proceed with `exp_012_hybrid_fullframe_tiles_640` as an inference-only paired
test using the same Exp10 checkpoint. Add one full-frame 960 pass to the exact
Exp11 tile set before the same global class-aware merge. This is a single
pipeline change intended to restore large-object/context predictions while
retaining the demonstrated tiny-object gain.

Pre-register preservation of AP-small and native tiny recall, meaningful
recovery of AP-large, and strict FalsePos/Dupe/cap/latency guardrails. Do not
simultaneously change tile size, overlap, NMS IoU, `max_det`, confidence floor,
threshold, checkpoint, or training. A full-frame-plus-tile ensemble may worsen
candidate pressure, so it is a test rather than an assumed fix.

Only after a context-preserving inference pipeline succeeds should tile-aware
or object-context-crop training be justified. A YOLO11s capacity experiment
remains useful for residual classification, but Exp11 indicates that effective
scale and context should be resolved before spending substantially more compute
on broad semantic capacity.

## Recommendation

Keep the Exp11 artifacts and implementation as strong evidence that VisDrone's
tiny-object problem is addressable through effective scale. Do not delete tiny
labels, claim statistical significance, or report Exp11 as a better trained
model—it is the same checkpoint under a heavier inference pipeline.

For a recall-priority application where roughly 15 FPS is acceptable, Exp11 is
a viable candidate after held-out threshold calibration. For a balanced
detector, retain Exp10 while testing the hybrid full-frame-plus-tile pipeline.
