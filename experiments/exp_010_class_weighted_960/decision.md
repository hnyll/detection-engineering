---
exp: "exp_010_class_weighted_960"
hypothesis_confirmed: false
outcome: inconclusive
next_experiment: "exp_011_tiled_inference_640"
---

# exp_010_class_weighted_960 — Decision

## Problem

Exp8's largest main TIDE opportunity was classification, with clean
rare-to-common errors such as truck-to-car and bicycle-to-motorcycle. Exp10
tested whether moderate inverse-frequency classification weighting addresses
that imbalance without increasing misses, false positives, or common-class
regressions.

## Result and preregistered decision

The intended mechanism did not pass. TIDE Cls moved only from 14.255 to 14.225
dAP, far short of the required 13.255. Mean bicycle/truck/bus AP improved from
0.22581 to 0.23520, missing its threshold by 0.00061. Overall mAP50-95 improved
0.00464 and mAP50 improved 0.00659, but both remained just below their
supporting thresholds. Only truck exceeded a +0.010 class gain.

Every guardrail passed: no class declined by more than 0.00210, AP-small rose,
Miss/Loc/Bkg and special FalsePos stayed within bounds, and inference cost was
unchanged. This makes the numerical direction mildly encouraging, but both
co-primary and every supporting rule failed.

Set `hypothesis_confirmed: false` and `outcome: inconclusive`. The right claim
is “small one-seed AP gains without evidence that class weighting fixed the
classification bottleneck,” not “Exp10 is a significant upgrade.”

## Causal interpretation

The runtime evidence confirms that `cls_pw=0.25` was applied as planned.
However, Exp10 was resumed twice, so exact augmentation/shuffle random streams
are not proven equivalent to Exp8's uninterrupted run. Combined with one seed,
this prevents assigning small sub-point gains confidently to weighting.

The result also weakens pure frequency imbalance as the main explanation for
Cls dAP. Exposure can still matter—truck improved 0.01506—but semantic
boundaries, visible pixels, context, and matching competition are at least as
important. Raising `cls_pw` would increase calibration/common-class risk
without evidence that the mechanism responds.

## Apparent background classifications

The confusion matrix's background row is not a learned background class. It is
unmatched GT at the display threshold. A threshold sweep shows that lowering
confidence from 0.25 to 0.05 raises recall from 0.6215 to 0.7430, but FP/image
rises from 18.67 to 95.11. At the full candidate floor, only about 16.5% of
unmatched GT have classification involvement; localization and irrecoverable
misses dominate.

Therefore:

- choose a deployment threshold only after defining an FP/recall budget;
- do not present a lower threshold as a better model;
- stop using class weights as the main fix for GT-background cases;
- target tiny-object feature scale and localization instead;
- correct omitted ignore-region handling separately before interpreting every
  prediction-background FP as a hallucination.

## Solution families considered

| Family | Evidence after Exp10 | Decision |
|---|---|---|
| stronger class weighting | Cls improves only 0.030 dAP | stop; not supported |
| confidence calibration | useful precision/recall tradeoff, no AP improvement | keep as deployment tuning |
| native-source tiles/context crops | tiny objects dominate Miss/Loc | select for fixed-checkpoint Exp11 diagnostic |
| YOLO11s full fine-tune | nano capacity may limit semantic representation | retain as a later capacity screen |
| generic stronger augmentation | current recipe already uses mosaic/HSV/scale/translation/flip | low priority; may erase tiny cues |
| ignore-aware evaluation | many apparent background FPs overlap omitted ignore regions | separate protocol correction |

## Recommendation

Do not run additional class-weight powers or extra seeds for this exact
treatment. Retain Exp10 as the inherited checkpoint for an inference-only
tiling diagnostic: keeping the weights fixed isolates whether magnifying native
source crops recovers the tiny GT objects that class weighting cannot.

If tiling demonstrates a large size-specific effect, preserve that mechanism
and solve its context/merge/latency tradeoffs before moving to heavier training.
If it does not, the clean next training screen is a stock YOLO11s full
fine-tune at 960 with `cls_pw=0`.
