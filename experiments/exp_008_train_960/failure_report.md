# exp_008_train_960 — Failure Analysis

Exp8 trains and evaluates YOLO11n at `imgsz=960`. Exp7 is the direct control:
it evaluates the 640-trained Exp6 checkpoint at the same 960 protocol. Both use
the six-class custom task, not official ten-class VisDrone Task 1.

The comparison identity is valid: effective protocol hash
`821306fe3a07ceac`, validation GT fingerprint `e1142868d4dc2e9d`, confidence
floor `0.001`, NMS IoU `0.7`, and `max_det=300` match. Exp8's evaluated
checkpoint SHA is `e1a30d15043d5955`. The configured initial-weight and
training-manifest hashes match the current files, but the trainer did not
runtime-attest those two input hashes. The dirty worktree also means git commit
`e10f4e5` does not pin uncommitted evaluator changes.

No Exp8 FiftyOne review was performed, so this report does not invent manual
failure counts or visual-condition prevalence. TIDE values are oracle AP
impacts, not counts, and its special categories overlap/reorganize the main
decomposition.

## Exp7/Exp8 metrics

| Metric | Exp7: train 640/eval 960 | Exp8: train 960/eval 960 | Delta | Interpretation |
|---|---:|---:|---:|---|
| mAP50-95 | 0.26661 | 0.28847 | +0.02186 | Material overall gain; primary passed |
| mAP50 | 0.44897 | 0.47411 | +0.02514 | Better detection/coarse localization |
| mAP75 | 0.27169 | 0.29178 | +0.02009 | Better stricter localization aggregate |
| AP-small | 0.17284 | 0.17472 | +0.00188 | Practically small; primary failed |
| AP-medium | 0.37210 | 0.41762 | +0.04552 | Strong gain |
| AP-large | 0.46163 | 0.62336 | +0.16173 | Very large recovery; few large examples |
| AR-max | 0.36598 | 0.38320 | +0.01722 | Supporting criterion passed |
| Mean latency | 10.77 ms | 10.40 ms | -0.37 ms | Same compute; treat as timing variation |
| FPS, batch 1 | 92.9 | 96.1 | +3.2 | Same timing caveat |
| Images at `max_det=300` | 476/548 | 463/548 | -13 images | Better, but 84.49% remain saturated |

mAP50-95 improved by 2.186 absolute points, or 8.20% relative. mAP50 and
mAP75 moved in the same direction, and every class improved, so this is a
practically meaningful seed-17 screen rather than a rounding-level change.
However, the comparison has one training seed per side, so its statistical
repeatability is unknown. The generated comparison correctly reports
`INSUFFICIENT-SEEDS`.

## Why small-object performance remains task-critical

COCO area bins applied to the six-class validation ground truth contain:

| Area bin | Ground-truth boxes | Share |
|---|---:|---:|
| Small (`area < 32^2`) | 26,586 | 68.59% |
| Medium (`32^2 <= area < 96^2`) | 11,105 | 28.65% |
| Large (`area >= 96^2`) | 1,068 | 2.76% |

Exp8 AP-small is 0.17472, versus 0.41762 medium and 0.62336 large. AP is not
the percentage of objects detected—it combines precision and recall across IoU
and confidence thresholds—but the scale gap is still large. Because nearly 69%
of validation annotations are small, weak small-object AP is a central
limitation for drone-view detection, not an edge case.

Exp8 therefore has two simultaneously true conclusions:

1. It is the strongest observed custom six-class model overall.
2. It remains unreliable on the object scale that dominates this dataset.

The 960 inference change in Exp7 produced most of the available small-object
gain (`0.11139 -> 0.17284`). Matching training to 960 added only 0.00188
AP-small. Exp8's additional improvement comes primarily from medium and large
objects, so 960 training is not evidence that the tiny-object problem is
solved. The very large AP-large delta should also be interpreted cautiously
because large boxes represent only 2.76% of annotations.

## Per-class checks

| Class | Exp7 AP50-95 | Exp8 AP50-95 | Delta |
|---|---:|---:|---:|
| person | 0.20608 | 0.22785 | +0.02177 |
| bicycle | 0.05032 | 0.07362 | +0.02330 |
| car | 0.58265 | 0.59229 | +0.00964 |
| truck | 0.19643 | 0.22348 | +0.02705 |
| bus | 0.36615 | 0.38033 | +0.01418 |
| motorcycle | 0.19802 | 0.23327 | +0.03525 |

All class guardrails pass, but bicycle remains extremely weak at 0.07362.
Class-level AP includes all sizes, so these improvements do not contradict the
nearly flat size-specific AP-small result.

## TIDE error impacts

Lower dAP is better. These values diagnose the AP opportunity remaining after
each model; they are nonlinear and must not be added into a total score.

| Error | Exp7 dAP | Exp8 dAP | Delta | Interpretation |
|---|---:|---:|---:|---|
| Cls | 13.895 | 14.255 | +0.360 | Slightly worse; still largest main mode |
| Loc | 5.086 | 4.863 | -0.223 | Small improvement; target failed |
| Both | 0.620 | 0.626 | +0.006 | Essentially unchanged |
| Dupe | 0.631 | 0.620 | -0.011 | Essentially unchanged |
| Bkg | 3.070 | 2.824 | -0.246 | Small improvement |
| Miss | 8.655 | 8.473 | -0.182 | Small improvement; target failed |
| Special FalsePos | 15.170 | 13.975 | -1.195 | Noticeable improvement; target passed |
| Special FalseNeg | 24.643 | 25.556 | +0.913 | Worse |

The TIDE movement is mostly modest. Exp8 meaningfully repairs Exp7's aggregate
false-positive tradeoff, but it does not materially reduce misses or
localization opportunity. Cls and special FalseNeg also move in the wrong
direction. This is compatible with higher mAP because TIDE decomposes residual
oracle-fix opportunity rather than acting as another overall model score.

## Automated classification bottleneck audit

An exact read-only audit reran tidecv's IoU-0.50 assignment over the Exp8 seed-17
GT and prediction files (`conf=0.001`, `max_det=300`). It is an automatic
protocol diagnostic, not a FiftyOne review or a count of deployment-visible
failures. Its machine-readable results and provenance are stored in
`artifacts/classification_bottleneck_audit_s17.json`. Of 38,759 GT objects,
29,752 were matched and 9,007 were unmatched.
Only 4,557 of those unmatched objects are TIDE `Miss`: the rest have at least
one classification and/or localization candidate that an oracle could repair.

| Exact assignment result | Count | Interpretation |
|---|---:|---|
| strict `Cls` detections | 6,230 | wrong class with IoU at least 0.50 |
| unique GT touched by `Cls` | 4,603 | includes GT that also has a correct prediction |
| unmatched GT with a `Cls` candidate | 1,479 | 16.42% of all 9,007 unmatched GT |
| unmatched GT with a localization-only candidate | 2,971 | box/class candidate exists but IoU is insufficient |
| unmatched GT with both candidate types | 238 | class and localization candidates coexist |
| irrecoverable `Miss` GT | 4,557 | no fixable candidate under this TIDE assignment |

This reconciles the apparently contradictory results: `Cls` can be the largest
macro-AP opportunity even though it explains only a minority of unmatched GT.
Rare-class errors carry disproportionate class-averaged AP impact, and 3,124 of
the 4,603 Cls-associated GT already have a separate true-positive detection.
Raw class-error box counts are therefore not a wrong-class miss rate.

### Clean directional class boundaries

Pair-specific oracle fixes identify a concentrated rare-to-common pattern:

| True class -> predicted class | Independent Cls dAP opportunity |
|---|---:|
| truck -> car | 3.352 |
| bicycle -> motorcycle | 2.346 |
| bus -> truck | 1.504 |
| bus -> car | 1.468 |

The total true-class classification opportunities are 4.454 dAP for truck,
3.042 for bus, and 2.952 for bicycle. At a cleaner diagnostic subset of
confidence at least 0.25 and IoU at least 0.75, the leading counts are 78
truck-to-car, 55 bicycle-to-motorcycle, and 50 motorcycle-to-car errors. Only 6
of 334 detections in that subset also overlap a GT of the predicted class, so
these are stronger class-boundary evidence than the raw person/motorcycle
counts. Rider/vehicle nesting and TIDE's class-before-duplicate ordering
contaminate many of the latter and prevent treating every one as a semantic
swap.

The repeated loader still contains 238,952 car instances versus 18,367 truck,
15,083 bicycle, and 9,505 bus instances. This supports a moderate class-weight
test, but it does not prove frequency imbalance is causal: object appearance,
context, label boundaries, and loss calibration remain alternative causes.

### Tiny scale and occlusion are mainly Miss/Loc conditions

Independent Cls opportunity is only 1.352 dAP for objects whose effective
shortest side at 960 is below 8 pixels, then rises to 4.235 at 8-16 pixels and
5.153 at 16-32 pixels. Ultra-tiny targets primarily disappear from the
candidate set instead of becoming well-localized wrong-class boxes:

| Native shortest side | GT | Unmatched GT | Irrecoverable Miss |
|---|---:|---:|---:|
| `<8 px` | 5,074 | 59.91% | 33.35% |
| `8-15 px` | 11,861 | 29.11% | 14.81% |
| `16-31 px` | 13,414 | 15.28% | 7.24% |
| `>=32 px` | 8,410 | 5.52% | 1.63% |

Original VisDrone occlusion metadata shows the same separation. Unmatched GT
rises from 16.68% with no occlusion to 26.92% with partial and 36.45% with
heavy occlusion; irrecoverable Miss rises from 7.51% to 14.18% and 20.09%.
Cls association stays approximately flat at 11-12%, while Loc association
rises from 28.32% to 39.13% and 45.43%. This supports size and occlusion as
Miss/Loc drivers, not as the main classification mechanism.

Most raw class-error detections are also low confidence: only 1,440/6,230 score
at least 0.25 and 435 score at least 0.50. Although 463/548 images hit
`max_det=300` at the 0.001 evaluation floor, none are capped at confidence 0.25.
The cap can affect low-ranked AP/recall, but it does not by itself explain the
clean high-confidence class pairs.

Finally, the derived GT omits 1,378 source `ignored-region` and 32 `others`
rows. At confidence at least 0.25, 1,364/3,473 TIDE Background detections are
covered by an omitted score-zero region over more than half their area. This is
an evaluation limitation for Bkg/FP interpretation, not an explanation for
the strict cross-class errors above. Ignore-aware evaluation should be a
separate protocol correction.

## Preregistered decision check

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Primary mAP50-95 | >=0.27661 | 0.28847 | Pass |
| Primary AP-small | >=0.18284 | 0.17472 | **Fail by 0.00812** |
| Supporting AR-max | >=0.37598 | 0.38320 | Pass |
| Supporting TIDE Miss | <=8.155 | 8.473 | Fail |
| Supporting TIDE Loc | <=4.586 | 4.863 | Fail |
| Supporting special FalsePos | <=14.170 | 13.975 | Pass |
| Guardrail AP-large | >=0.44163 | 0.62336 | Pass |
| Guardrail TIDE Cls | <=14.895 | 14.255 | Pass |
| Guardrail special FalsePos | <=16.170 | 13.975 | Pass |
| Guardrail mean latency | <=13.50 ms | 10.40 ms | Pass |
| Per-class AP decline | no class worse by >0.010 | all six improved | Pass |

Both primary criteria were required. AP-small failed, so the composite
hypothesis is not confirmed even though mAP passed, exactly two supporting
criteria passed, and every guardrail passed. The practical outcome remains
positive.

## Training diagnostics

The run completed normally for 100 epochs in 23,717 seconds (6 h 35 min).
Ultralytics selected epoch 76 as `best.pt`, with internal validation mAP50-95
0.29567. Epoch 100 ended at 0.29179, a decline of 0.00388. Training losses kept
falling while validation metrics plateaued around epochs 70-90, indicating mild
late overfit rather than a failed or severely overtrained run. The final
ten-epoch training-loss shift is also influenced by `close_mosaic=10`.
Evaluation uses `best.pt`, so the late decline does not invalidate the fixed
evaluation result. GPU VRAM was not recorded; the state file's 4,058 MB value is
process resident RAM, not VRAM.

## Root-cause hypotheses after Exp8

| Hypothesis | Supporting evidence | Confidence (0-1) |
|---|---|---:|
| small-object candidate/search limitation, amplified by occlusion | native `<8 px` targets are 59.91% unmatched; unmatched risk rises from 16.68% with no occlusion to 36.45% with heavy occlusion | 0.95 |
| matched 960 training mainly helps medium/large scale behavior | AP-medium gained 0.04552 and AP-large 0.16173 while AP-small was flat | 0.85 |
| weak directional class boundaries | truck-to-car, bicycle-to-motorcycle, and bus-to-road-vehicle fixes provide 3.352, 2.346, and about 1.5 dAP each | 0.90 |
| residual rare-to-common exposure imbalance contributes to Cls | the repeated loader still has 238,952 car instances versus 18,367 truck, 15,083 bicycle, and 9,505 bus; causal contribution is untested | 0.70 |
| omitted ignore regions contaminate Bkg/FP diagnosis | 1,364/3,473 Bkg detections at confidence >=0.25 substantially overlap omitted score-zero regions | 0.85 |
| the low-confidence detection cap amplifies residual recall/AP errors | 463/548 images are capped at confidence 0.001, but none are capped at 0.25 | 0.40 |
| 640/960 scale mismatch was the main tiny-object limiter | Exp8 directly treated it but added only 0.00188 AP-small | 0.20 |
| NMS duplication is the main limiter | Dupe impact is only 0.620 dAP | 0.10 |

## Top-3 remaining main failure modes

1. Class-decision errors (`Cls` 14.255 dAP), led by clean truck/car,
   bicycle/motorcycle, and bus/road-vehicle directions; raw rider/person pairs
   include overlap/matching contamination.
2. Missed objects (`Miss` 8.473 dAP): 9,007 GT are unmatched and 4,557 are
   irrecoverable TIDE Misses, concentrated among tiny and occluded targets.
3. Poor localization (`Loc` 4.863 dAP): 2,971 unmatched GT have a
   localization-only candidate and 238 have both class and localization
   candidates.

Size and occlusion are contributing conditions within Miss/Loc, not replacement
top-level TIDE modes. Special FalseNeg is overlapping aggregate context and is
not an additional fourth category.

## Interpretation

Exp8 is a meaningful overall improvement at the same 960 deployment cost and
is the best observed six-class candidate. It should be adopted provisionally if
a one-seed engineering screen is sufficient. It should not be described as
statistically significant, an official VisDrone Task 1 result, or a solution to
tiny-object detection. Further full-image resolution increases are not the
first intervention supported by the flat AP-small response.
