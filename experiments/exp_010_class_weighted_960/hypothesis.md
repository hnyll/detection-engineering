# exp_010_class_weighted_960 — Hypothesis

## Question

Does moderate inverse-frequency weighting of the classification loss reduce the
clean rare-to-common class confusions left by Exp8 without trading them for
misses, false positives, or common-class regressions?

## Prediction (written before training)

Exp8's repeated training view is still highly imbalanced: car contributes
238,952 instances, compared with 18,367 truck, 15,083 bicycle, and 9,505 bus
instances. Its strongest clean wrong-class directions are truck to car,
bicycle to motorcycle, and bus to truck/car. With `cls_pw=0.25`, Ultralytics
should moderately upweight rare-class BCE terms and downweight common-class BCE
terms.

The screen is successful only if **both** co-primary criteria pass:

1. TIDE `Cls` falls from 14.255 to at most **13.255 dAP**.
2. Mean AP50-95 for bicycle, truck, and bus rises from 0.22581 to at least
   **0.23581**.

At least one supporting criterion must also pass:

- overall mAP50-95 is at least **0.29347** (+0.005 versus Exp8);
- mAP50 is at least **0.48411** (+0.010);
- at least two of bicycle, truck, and bus individually improve by 0.010 AP50-95.

All guardrails must pass:

- overall mAP50-95 is at least 0.28347;
- no class AP50-95 declines by more than 0.010;
- AP-small is at least 0.16972;
- TIDE `Miss` is at most 8.973, `Loc` at most 5.363, and `Bkg` at most 3.324;
- special FalsePos is at most 14.975 and special FalseNeg at most 26.556;
- mean batch-1 latency is at most 12.50 ms on the recorded machine.

These are practical one-seed screening thresholds, not confidence intervals.
If the screen passes, matched additional Exp8 and Exp10 seeds are needed before
claiming repeatability.

## Background evidence

An exact automated audit of Exp8's fixed prediction set found 6,230 TIDE
ClassError detections, but only 1,479 of 9,007 unmatched GT objects had a
strict wrong-class candidate at IoU at least 0.50. Classification is therefore
important without explaining most misses. Pair-specific oracle tests identify
truck to car (3.352 dAP), bicycle to motorcycle (2.346), bus to truck (1.504),
and bus to car (1.468) as the clearest directional class boundaries.

Ultra-tiny objects are mainly a separate miss/localization problem. The
classification oracle impact peaks in the 8-32 pixel effective-short-side
range, while native objects below 8 pixels have a 59.91% unmatched rate.
Occlusion likewise increases miss/localization rates but not class-error
association. This experiment therefore targets class imbalance only; it is not
expected to solve tiny-object recall or occlusion.

Ultralytics 8.4.65 computes `normalize_mean((1/count)^cls_pw)` from the actual
repeated training loader and applies each weight to the full per-class BCE,
including negative terms. The preregistered `cls_pw=0.25` produces:

| Class | Loader instances | Expected weight |
|---|---:|---:|
| person | 146,233 | 0.697288 |
| bicycle | 15,083 | 1.230415 |
| car | 238,952 | 0.616731 |
| truck | 18,367 | 1.171289 |
| bus | 9,505 | 1.380974 |
| motorcycle | 51,923 | 0.903303 |

Because negative BCE terms are weighted too, calibration and false-positive
behavior are explicit guardrails rather than assumed benefits.

## Baseline

Direct control: `exp_008_train_960`, seed 17. Both experiments train fresh from
the same stock `yolo11n.pt`, use the exact same six-class repeat-factor
manifest, and evaluate at the same 960 protocol.

## Causal variable changed

`train.cls_pw`: **0.0 -> 0.25**.

## Fixed inputs and excluded changes

Model, initialization, seed, manifest, taxonomy, image size, epochs, batch,
`nbs`, augmentations, optimizer defaults, and evaluation protocol remain fixed.
Do not also change the global `cls` gain, sampler, architecture, confidence
floor, NMS, or `max_det`. Exp10 starts from stock weights rather than continuing
from Exp8 so prior training is not an extra variable.

The existing Exp9 is an abandoned analysis-only manual signal audit. It is
retained for provenance and is not part of the detector checkpoint lineage.
