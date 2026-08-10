---
exp: "exp_008_train_960"
hypothesis_confirmed: false
outcome: positive
next_experiment: "exp_010_class_weighted_960"
---

# exp_008_train_960 — Decision

## Problem

Exp7 showed that the 640-trained Exp6 checkpoint performs substantially better
at 960 inference, but false-positive impact increased and AP-large declined.
Exp8 tested whether training and checkpoint selection at 960 improve that
accuracy/precision tradeoff and provide additional small-object benefit.

## Result

At the identical 960 evaluation protocol, Exp8 improves mAP50-95 from 0.26661
to 0.28847 (+0.02186), mAP50 from 0.44897 to 0.47411, mAP75 from 0.27169 to
0.29178, and AR-max from 0.36598 to 0.38320. All six classes improve. Special
FalsePos impact decreases from 15.170 to 13.975 dAP, repairing Exp7's principal
guardrail failure.

The improvement does not come primarily from tiny objects. AP-small moves only
from 0.17284 to 0.17472 (+0.00188), while AP-medium rises 0.04552 and AP-large
rises 0.16173. Miss and Loc improve by only 0.182 and 0.223 dAP; Cls worsens by
0.360 and special FalseNeg worsens by 0.913.

## Preregistered outcome

The mAP primary criterion passes, but the AP-small primary criterion fails by
0.00812. Supporting AR and special FalsePos pass; Miss and Loc fail. Every
guardrail passes, including every per-class floor.

Because both primary criteria were required, `hypothesis_confirmed` is
**false** under the written rule. The experiment outcome is **positive**:
Exp8 gains 2.186 absolute mAP50-95 points at the same inference architecture,
resolution, and theoretical compute, improves every class, and reduces
false-positive impact.

## Automated bottleneck check

An exact tidecv assignment audit separates the remaining branches. Only 1,479
of 9,007 unmatched GT objects have a strict wrong-class candidate, while 4,557
are irrecoverable Misses and 3,209 have localization involvement. Tiny scale and
occlusion strongly increase Miss/Loc rates but not class-error association.

Classification still has high macro-AP leverage through concentrated rare-class
boundaries: independent oracle fixes yield 3.352 dAP for truck-to-car, 2.346 for
bicycle-to-motorcycle, 1.504 for bus-to-truck, and 1.468 for bus-to-car. The
repeated loader remains dominated by car instances. This supports one moderate
class-weighting screen, while making clear that it is not a general solution to
tiny-object misses.

## Task-level interpretation

Small-object performance remains a major limitation. Under COCO area bins,
26,586 of 38,759 validation annotations (68.59%) are small, yet Exp8 AP-small
is only 0.17472, versus 0.41762 medium and 0.62336 large. AP is not a detection
rate, but this large scale gap shows that the model is weakest on the majority
object regime in these drone images.

Therefore Exp8 is both the best observed custom six-class model and still a
weak tiny-object detector. If the application requires reliable detection of
distant people and vehicles, the model is not “good enough” merely because its
aggregate mAP improved. If the application tolerates missed tiny objects and
prioritizes medium/large targets, Exp8 is a reasonable experimental candidate.

These metrics are for a custom six-class taxonomy. They cannot be presented as
an official ten-class VisDrone Task 1 benchmark score.

## Causal interpretation

Compared with Exp7, Exp8 changes training resolution from 640 to 960 while
evaluation remains 960. It starts fresh from stock `yolo11n.pt` and reuses the
same six-class taxonomy, exact repeat-factor manifest, seed, architecture, and
epoch budget. Batch decreases from 16 to 4 for VRAM safety; `nbs=64` preserves
the nominal accumulated batch, but microbatch and BatchNorm behavior remain a
documented confound.

The evidence supports resolution-matched training as an overall accuracy and
false-positive calibration improvement, mainly for medium and large objects.
It does not support training-resolution mismatch as the main remaining cause of
small-object errors.

## Training quality

Training completed normally in 6 h 35 min. Internal validation mAP50-95 peaked
at epoch 76 (`best.pt`) and declined 0.00388 by epoch 100. This is a mild late
plateau/overfit pattern, not severe overtraining; evaluation correctly uses the
best checkpoint.

## Tradeoffs and remaining limitations

- AP-small remains low on the majority object-size regime.
- Bicycle remains weakest at 0.07362 AP50-95.
- Main TIDE errors remain Cls 14.255, Miss 8.473, and Loc 4.863 dAP.
- Special FalseNeg worsened even though aggregate AR increased.
- `max_det=300` remains saturated on 463 of 548 images at confidence 0.001,
  although no image is capped at confidence 0.25; this is mainly a low-ranked
  candidate diagnostic.
- Automated TIDE assignments are protocol-specific diagnostics, not manual
  prevalence counts.
- Omitted source ignore regions contaminate absolute Bkg/FP interpretation;
  correcting them requires a distinct evaluation protocol.
- The very large AP-large gain is based on a bin containing only 2.76% of GT
  boxes and may be less stable than the aggregate result.
- One seed does not establish statistical significance or repeatability.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Classification loss | Moderate inverse-frequency weighting (`cls_pw=0.25`) at 960 | Select for Exp10 as the next isolated screen |
| Further full-image resizing | Train/evaluate above 960 | Defer; matched 960 training barely changed AP-small |
| Native-resolution tiling/crops | Overlapping tiles or object-context crops | Most direct next family for preserving tiny-object source detail; test separately |
| Model capacity | YOLO11s at 960 | Valid semantic-capacity experiment, but increases training/inference cost |
| Detection cap | Test `max_det > 300` with fixed Exp8 weights | Cheap diagnostic for saturation; separate inference protocol |
| Confidence calibration | Select an operating threshold for a target precision/FP budget | Do after model selection; does not improve COCO AP |
| NMS tuning | Change NMS IoU | Low priority because Dupe impact is only 0.620 dAP |
| Tiny-label removal | Remove small annotations | Reject unless the application also excludes those objects during evaluation |

## Recommendation

Adopt Exp8 provisionally as the current best custom six-class checkpoint, while
documenting that it remains weak on the dominant small-object regime. Do not
claim statistical significance or official VisDrone Task 1 performance.

Proceed with `exp_010_class_weighted_960` as a one-variable seed-17 screen:
start fresh from the same stock checkpoint, keep Exp8's repeat manifest and 960
protocol fixed, and change only `cls_pw` from 0 to 0.25. Require both a material
Cls reduction and rare-class AP gain without Miss/FP/common-class regressions.
If it passes, run matched additional Exp8 and Exp10 seeds before adoption. If
it fails, stop frequency-weight tuning and test semantic capacity or targeted
object-context crops. Native-detail/tiling work remains the separate branch for
tiny-object misses; ignore-aware rescoring remains an evaluation correction.
