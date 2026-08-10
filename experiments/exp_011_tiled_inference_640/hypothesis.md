# exp_011_tiled_inference_640 — Hypothesis

## Question

Does inference on overlapping native-source tiles reduce Exp10's tiny-object
miss/localization failures when the checkpoint, labels, confidence floor, NMS
IoU, final detection budget, and model input size remain fixed?

## Prediction (written before evaluation)

Exp10's native objects below 8 pixels on their shortest side are unmatched at
59.28%, compared with 5.60% for objects at least 32 pixels. Exp11 will crop the
source into 640x640 windows with 20% overlap and infer every crop at 960. This
increases the effective feature size of tiny objects without retraining.

Both mechanism co-primary criteria must pass:

1. AP-small increases from 0.17903 to at least **0.18903** (+0.010).
2. Native-shortest-side `<8 px` recall increases from 0.4072 to at least
   **0.4572** (+0.050).

At least two supporting criteria must pass:

- native `8-15 px` recall is at least 0.7374 (+0.030);
- AR-max is at least 0.39521 (+0.010);
- TIDE Miss is at most 7.800 (-0.500 dAP);
- TIDE Loc is at most 4.600 (-0.250 dAP).

All accuracy guardrails must pass:

- mAP50-95 is at least 0.28811 (no regression greater than 0.005);
- no class AP50-95 declines by more than 0.010;
- AP-medium is at least 0.41011 and AP-large at least 0.55783;
- TIDE Bkg is at most 3.813, Dupe at most 1.104, special FalsePos at most
  15.811, and special FalseNeg at most 26.865;
- no invalid/out-of-image box is emitted and no more than 95% of images reach
  the unchanged global `max_det=300` cap.

Deployment is evaluated separately: end-to-end mean latency, including image
decode, slicing, every tile pass, and global merge, must not exceed 105.5
ms/image (10x Exp10, approximately 9.5 FPS). If the accuracy mechanism passes
but latency fails, record "mechanism supported, deployment negative" rather
than changing the accuracy conclusion.

These are engineering thresholds on one deterministic inherited checkpoint,
not confidence intervals or an unbiased benchmark claim.

## Background evidence

At Exp10's full 0.001 candidate floor, 8,997 GT objects remain unmatched.
Tiny objects dominate: native `<8 px` targets are 59.28% unmatched, and tiny
plus occluded objects account for most Miss/Loc involvement. Class weighting
barely changed TIDE Cls, so another loss-weight adjustment is not the treatment
here. Tiling is tested before tile-aware training because the unchanged
checkpoint provides a cheap direct test of whether effective feature scale is
causal.

## Baseline

Direct paired control: `exp_010_class_weighted_960`, seed 17, full-frame
inference. Exp11 inherits its exact `best.pt` checkpoint SHA
`d078cf0e25577202` and the same GT fingerprint `e1142868d4dc2e9d`.

## Causal variable changed

Prediction pipeline: one full-frame 960 pass becomes pure 640x640 native-source
tiles with 20% overlap, each passed at `imgsz=960`, followed by global
class-aware NMS at IoU 0.70 and the same top-300 final budget.

## Fixed inputs and excluded changes

Checkpoint, dataset, taxonomy, validation split, model input size, confidence
floor 0.001, per-tile NMS IoU 0.70, class-aware behavior, and final
`max_det=300` remain fixed. Exp11 performs no training, TTA, full-image
ensemble, confidence tuning, ignore-region correction, `max_det` increase, or
NMS sweep.

Multiple passes, crop boundaries, coordinate remapping, and the global merge
are inseparable implementation consequences of tiling. Tiling magnifies
available source pixels; it cannot create information absent from the image.
It may increase boundary/duplicate/background predictions, truncate large
objects, and multiply latency. The current GT intentionally retains the same
omitted-ignore limitation as Exp10 for causal comparability.
