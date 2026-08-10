# exp_010_class_weighted_960 — Failure Analysis

Exp10 repeats the Exp8 YOLO11n/960 training recipe and changes only
Ultralytics' inverse-frequency classification weight power from `cls_pw=0` to
`0.25`. Both start fresh from the same stock checkpoint and use the same
six-class repeat-factor manifest and fixed 960 evaluation protocol.

No Exp10 FiftyOne review was performed, so this report does not invent manual
counts. TIDE values are oracle AP impacts (dAP), not counts; its special
FalsePos/FalseNeg partition overlaps the main categories.

## Runtime treatment verification

| Check | Expected | Observed | Result |
|---|---:|---:|---|
| `cls_pw` | 0.25 | 0.25 | pass |
| training entries | 8,716 | 8,716 | pass |
| person count / weight | 146,233 / 0.697288 | 146,231 / 0.697289 | pass; duplicate-label normalization difference |
| bicycle count / weight | 15,083 / 1.230415 | 15,083 / 1.230413 | pass |
| car count / weight | 238,952 / 0.616731 | 238,949 / 0.616732 | pass; duplicate-label normalization difference |
| truck count / weight | 18,367 / 1.171289 | 18,367 / 1.171287 | pass |
| bus count / weight | 9,505 / 1.380974 | 9,505 / 1.380972 | pass |
| motorcycle count / weight | 51,923 / 0.903303 | 51,922 / 0.903306 | pass; duplicate-label normalization difference |
| initial-weight SHA-256 | `0ebbc80d...7644ee1` | exact match | pass |
| train-manifest SHA-256 | `2b3a16c5...1ea74d3` | exact match | pass |

The runtime callback verified all six computed weights; maximum deviation from
the preregistered rounded values was `3.19e-6`. Ultralytics applies these
weights to positive and negative per-class BCE terms, so this is not equivalent
to merely repeating rare positive objects.

## Preregistered outcome

| Criterion | Exp8 | Required Exp10 | Exp10 | Delta | Result |
|---|---:|---:|---:|---:|---|
| TIDE Cls, co-primary | 14.255 | <=13.255 | 14.225 | -0.030 | fail by 0.970 |
| bicycle/truck/bus mean AP, co-primary | 0.22581 | >=0.23581 | 0.23520 | +0.00939 | fail by 0.00061 |
| overall mAP50-95, supporting | 0.28847 | >=0.29347 | 0.29311 | +0.00464 | fail by 0.00036 |
| mAP50, supporting | 0.47411 | >=0.48411 | 0.48070 | +0.00659 | fail by 0.00341 |
| at least 2/3 target classes gain >=0.010 | — | 2/3 | 1/3 | — | fail; truck only |
| overall mAP guardrail | 0.28847 | >=0.28347 | 0.29311 | +0.00464 | pass |
| AP-small guardrail | 0.17472 | >=0.16972 | 0.17903 | +0.00431 | pass |
| TIDE Miss guardrail | 8.473 | <=8.973 | 8.300 | -0.173 | pass |
| TIDE Loc guardrail | 4.863 | <=5.363 | 4.850 | -0.013 | pass |
| TIDE Bkg guardrail | 2.824 | <=3.324 | 2.813 | -0.011 | pass |
| special FalsePos guardrail | 13.975 | <=14.975 | 13.811 | -0.164 | pass |
| special FalseNeg guardrail | 25.556 | <=26.556 | 25.865 | +0.309 | pass |
| mean latency guardrail | 10.40 ms | <=12.50 ms | 10.55 ms | +0.15 ms | pass |

Both co-primary criteria and every supporting criterion failed, although the
rare-class mean and overall mAP thresholds were narrowly missed. Every
guardrail passed. The strict hypothesis is false; the practical result is a
small numerical gain but inconclusive evidence for the intended classification
mechanism.

## Accuracy outcomes

| Metric | Exp8 | Exp10 | Delta |
|---|---:|---:|---:|
| mAP50-95 | 0.28847 | 0.29311 | +0.00464 |
| mAP50 | 0.47411 | 0.48070 | +0.00659 |
| mAP75 | 0.29178 | 0.29885 | +0.00707 |
| AP-small | 0.17472 | 0.17903 | +0.00431 |
| AP-medium | 0.41762 | 0.42011 | +0.00249 |
| AP-large | 0.62336 | 0.58783 | -0.03553 |
| AR-max | 0.38320 | 0.38521 | +0.00201 |

The aggregate gains are directionally positive but small relative to ordinary
single-seed training variability, which is not measured here. AP-large moves
in the opposite direction; large objects are only 2.76% of this validation GT,
so the regression is worth retaining without treating it as a stable estimate.

## Per-class AP50-95

| Class | Exp8 | Exp10 | Delta | Guardrail |
|---|---:|---:|---:|---|
| person | 0.22785 | 0.22575 | -0.00210 | pass |
| bicycle | 0.07362 | 0.07835 | +0.00473 | pass |
| car | 0.59229 | 0.59110 | -0.00119 | pass |
| truck | 0.22348 | 0.23854 | +0.01506 | pass |
| bus | 0.38033 | 0.38872 | +0.00839 | pass |
| motorcycle | 0.23327 | 0.23618 | +0.00291 | pass |

Truck shows the clearest intended benefit. Bicycle and bus move positively but
below the preregistered effect size; common person and car decline slightly.
No class violates its safety floor.

## TIDE outcomes

| Error | Exp8 dAP | Exp10 dAP | Delta | Interpretation |
|---|---:|---:|---:|---|
| Cls | 14.255 | 14.225 | -0.030 | effectively unchanged; primary failed |
| Loc | 4.863 | 4.850 | -0.013 | effectively unchanged |
| Both | 0.626 | 0.647 | +0.021 | effectively unchanged/slightly worse |
| Dupe | 0.620 | 0.604 | -0.016 | effectively unchanged |
| Bkg | 2.824 | 2.813 | -0.011 | effectively unchanged |
| Miss | 8.473 | 8.300 | -0.173 | small directional improvement |
| special FalsePos | 13.975 | 13.811 | -0.164 | small directional improvement |
| special FalseNeg | 25.556 | 25.865 | +0.309 | worse |

AP can rise while residual TIDE changes little. AP uses confidence-ranked,
class-averaged precision/recall across IoUs; TIDE asks how much AP this new
candidate set could gain from an oracle correction. A ranking/calibration gain
or several modest rare-class gains can raise AP without materially shrinking
the residual Cls oracle ceiling.

## What the confusion-matrix “background” row means

YOLO does not output a foreground class named background. In the Ultralytics
confusion matrix, the bottom row means a GT object was unmatched at the plot's
display operating point (approximately confidence 0.25 and IoU above 0.45).
It can mean low confidence, weak localization, a wrong class, one-to-one match
competition in a crowd, or a true absent candidate.

The fixed-prediction threshold sweep makes the main tradeoff visible at
same-class IoU 0.50:

| Confidence | Micro precision | Micro recall | F1 | FP/image | Unmatched GT |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.3559 | 0.7430 | 0.4813 | 95.11 | 9,960 |
| 0.10 | 0.4611 | 0.7137 | 0.5603 | 58.99 | 11,095 |
| 0.25 | 0.7019 | 0.6215 | 0.6592 | 18.67 | 14,671 |
| 0.30 | 0.7622 | 0.5902 | 0.6652 | 13.02 | 15,885 |

Lowering confidence recovers many GT objects, proving that much of the display
background row is “model response below the display threshold,” not total lack
of signal. It also increases false alarms sharply. Confidence calibration is
therefore a deployment precision/recall choice, not a learned-model fix.

At the full 0.001 candidate floor, an exact TIDE-style assignment left 8,997
GT unmatched: 1,239 had a classification-only candidate, 3,051 a
localization-only candidate, 243 both candidate types, and 4,464 were TIDE
Miss. Most unresolved GT-background behavior is consequently Miss/Loc rather
than something another class-weight adjustment should solve. Tiny and occluded
objects are especially concentrated in those branches.

The omitted original VisDrone ignore regions are a separate evaluator issue:
they can make real predictions appear as background false positives in the
rightmost matrix column, but they cannot explain retained GT objects becoming
unmatched in the bottom row.

## Training and provenance caveats

The run completed all 100 epochs and evaluation uses `best.pt` from epoch 78.
Internal validation mAP50-95 peaked at 0.30226; the final epoch was 0.29860,
consistent with a mild late plateau rather than severe overtraining.

Operationally, training used an initial segment and two resume commands.
Ultralytics restored model, optimizer, scheduler, and EMA state, but exact
Python/NumPy/Torch/dataloader random streams are not proven identical to an
uninterrupted run. This makes Exp10 a valid engineering checkpoint but weakens
the claim that `cls_pw` was the only realized stochastic difference from Exp8.
One seed also cannot establish repeatability or statistical significance.

## Top-3 remaining main failure modes

1. Classification decision errors (`Cls` 14.225 dAP), essentially unchanged
   by moderate inverse-frequency loss weighting.
2. Missed objects (`Miss` 8.300 dAP), concentrated among tiny and occluded GT.
3. Poor localization (`Loc` 4.850 dAP), a major contributor to apparent
   GT-background cases at a normal display threshold.

## Conclusion

Moderate class weighting produced small AP gains—most clearly for truck—and
violated no guardrail. It did not materially reduce the targeted TIDE
classification opportunity. Do not increase `cls_pw` or claim that frequency
weighting solved semantic confusion. Preserve Exp10 as a useful exploratory
checkpoint, but move the next isolated test upstream to effective feature
scale/context or model capacity.
