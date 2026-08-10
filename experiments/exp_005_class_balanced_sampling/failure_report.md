# exp_005_class_balanced_sampling — Failure Analysis

This analysis compares Exp5 seed 17 with Exp1 seed 17 under fixed evaluation
protocol `a9b8207b3cc7a070`. No Exp5 FiftyOne review/export was performed, so
the report does not invent raw error counts or class-confusion pairs. TIDE
values are oracle AP impacts (dAP), not counts; its special false-positive and
false-negative categories overlap the main error decomposition and must not be
added to it.

## Preregistered outcome

| Criterion | Exp1 s17 | Exp5 s17 | Delta | Required | Result |
|---|---:|---:|---:|---:|---|
| rare-class mean AP50-95 | 0.12661 | 0.12849 | +0.00189 | at least +0.015 | fail |
| overall mAP50-95 | 0.17026 | 0.17265 | +0.00239 | at least +0.005 | fail |
| TIDE classification dAP | 23.065 | 23.250 | +0.185 (worse) | decrease by at least 1.5 | fail |
| car AP50-95 guardrail | 0.49711 | 0.49916 | +0.00205 | decline no more than 0.010 | pass |

The three intended benefits failed their thresholds; only the guardrail
passed. The hypothesis is therefore not confirmed.

## Outcomes

| Outcome | Exp5 impact | Delta vs Exp1 s17 | Notes |
|---|---:|---:|---|
| Wrong class | Cls dAP 23.250 | +0.185 (worse) | Largest individual TIDE error type; no manual class-pair count |
| Missed object | Miss dAP 8.243 | -0.116 (better) | Main TIDE decomposition |
| Poor localization | Loc dAP 4.673 | +0.025 (worse) | Third-largest individual TIDE error type |
| Class + localization | Both dAP 0.625 | -0.031 (better) | Neither a pure class nor pure localization error |
| Duplicate | Dupe dAP 0.364 | -0.028 (better) | Small impact; does not support NMS as a priority |
| Background prediction | Bkg dAP 1.795 | +0.045 (worse) | Main TIDE decomposition |
| False positive, aggregate | special dAP 13.735 | -0.326 (better) | Special TIDE aggregate, not a raw FP count |
| False negative, aggregate | special dAP 32.625 | +0.453 (worse) | Special TIDE aggregate, not a raw FN count |

## Per-class checks

| Class | Exp1 s17 AP50-95 | Exp5 s17 AP50-95 | Delta |
|---|---:|---:|---:|
| bicycle | 0.02953 | 0.03485 | +0.00532 |
| truck | 0.16377 | 0.16377 | +0.00000 |
| tricycle | 0.10084 | 0.10118 | +0.00034 |
| awning-tricycle | 0.04319 | 0.04331 | +0.00012 |
| bus | 0.29571 | 0.29936 | +0.00365 |
| **target-class mean** | **0.12661** | **0.12849** | **+0.00189** |
| car (guardrail) | 0.49711 | 0.49916 | +0.00205 |

Bicycle improved most among the targeted classes, but truck was unchanged and
the remaining gains were very small. These changes do not constitute the
predicted rare-class improvement.

## Conditions present in reviewed failures

| Condition | Exp5 reviewed count | Evidence boundary |
|---|---:|---|
| tiny | not collected | AP-small is an aggregate size metric, not a manual failure count |
| crowded | not collected | No Exp5 saved review view |
| occluded | not collected | Prior Exp1 observations cannot be reported as Exp5 counts |
| blur | not collected | Prior Exp1 observations cannot be reported as Exp5 counts |
| low-light | not collected | Prior Exp1 observations cannot be reported as Exp5 counts |
| truncation | not collected | Prior Exp1 observations cannot be reported as Exp5 counts |

## Root-cause hypotheses

| Hypothesis | Supporting evidence | Confidence (0–1) |
|---|---|---:|
| image-level co-occurrence diluted the intended rebalance | The manifest expanded from 6,471 to 10,095 entries (1.56x). Target-class instances increased by about 1.75x, but car instances also increased by 1.62x because repeated images contain several classes. The aggregate rare-to-car exposure ratio improved only about 8.4%. | 0.85 |
| class-frequency imbalance alone is the primary cause | Exposure changed substantially while target-class mean AP increased only 0.00189 and Cls dAP worsened. This argues against frequency alone, although the broad sampler was not a perfect test of exact class balance. | 0.25 |
| intrinsic visual or label-taxonomy ambiguity | Cls dAP remained dominant at 23.250 despite additional exposure. Prior Exp1 review found visually ambiguous class boundaries, but Exp5 has no pair-level manual review, so the exact pairs remain provisional. | 0.70 |
| insufficient pixels or context for small objects | AP-small remained 0.09019 versus 0.26691 for medium and 0.41105 for large objects, while special FalseNeg dAP remained high at 32.625. | 0.70 |
| oversampling overfit | Trainer validation peaked at epoch 83 and declined mildly by epoch 100 while training loss fell, but Exp1 showed a similar late pattern and evaluation uses `best.pt`. There is no evidence of unusually severe Exp5 overtraining. | 0.20 |
| NMS suppression is a primary cause | Dupe dAP was only 0.364. Post-NMS TIDE does not directly diagnose assignment, but duplicate impact gives no reason to prioritize NMS. | 0.10 |

## Top-3 measured failure modes

1. Wrong-class or insufficient-semantic-evidence failures (`Cls dAP 23.250`).
2. Missed objects, with a persistent small-object gap (`Miss dAP 8.243`,
   `AP-small 0.09019`, and special `FalseNeg dAP 32.625`).
3. Box localization failures (`Loc dAP 4.673`), materially smaller than
   classification and misses but larger than background and duplicate impact.

## Interpretation

Exp5 produced small improvements in mAP50 (+0.00523), AP-small (+0.00403),
AP-medium (+0.00419), and AR-max (+0.00148), but essentially no mAP75 change
(-0.00005). AP-large declined by 0.03463, although a single seed is not enough
to establish whether that regression is repeatable.

The correct conclusion is narrow: this exact repeat-factor image sampler did
not deliver the preregistered rare-class or classification benefit. It does not
prove that all balancing strategies are ineffective. Exp5 has one seed, and no
manual Exp5 review was performed, so causal claims about individual confusion
pairs or failure-condition prevalence remain unsupported.
