# exp_004_p2_head_640 — Failure Analysis

Use the original ten-class VisDrone Task 1 predictions. Keep automatic TIDE
impacts separate from any exploratory FiftyOne observations; manual tags are not
dataset-wide prevalence estimates.

## Outcomes (counts from FiftyOne evaluation / TIDE)
| Outcome | Count / impact | Notes |
|---|---:|---|
| TP | not collected | No Exp4 FiftyOne review/export was performed |
| FP (all unmatched predictions) | FalsePos dAP 16.802 | TIDE impact, not a raw count; conf=0.001 |
| FN (unmatched GT) | FalseNeg dAP 28.215 | TIDE impact, not a raw count |
| Duplicate | Dupe dAP 0.667 | Raw count not collected |
| Wrong class | Cls dAP 24.866 | Largest individual TIDE error impact |
| Poor localization | Loc dAP 2.909 | Smaller impact than classification and misses |

## Conditions present in reviewed failures
| Condition | # reviewed labels | Notes |
|---|---:|---|
| tiny | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |
| crowded | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |
| occluded | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |
| blur | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |
| low-light | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |
| truncation | not reviewed | Exp1 observations may motivate hypotheses but are not Exp4 counts |

## Root-cause hypotheses
| Hypothesis | Supporting evidence | Confidence (0–1) |
|---|---|---:|
| classification/label ambiguity and imbalance | Cls dAP 24.866 is much larger than Loc 2.909 and Miss 8.016. Prior review identified visually ambiguous car/van/truck/bus, pedestrian/people, and bicycle-family boundaries. Training annotations are also imbalanced: car 144,866 versus awning-tricycle 3,246, tricycle 4,812, and bus 5,926. Exp4 has no class-pair count, so the pair-level claim remains provisional. | 0.75 |
| limited information in tiny objects | AP-small 0.09465 remains far below AP-medium 0.24558 and AP-large 0.34997. P2 increased AP-small and AR, showing that scale matters, but a finer feature grid cannot recover class evidence absent from a few source pixels. | 0.65 |
| shallow/noisy P2 predictions | Adding P2 increases prediction locations at 640px from 8,400 to 34,000. FalsePos dAP increased from Exp1's 14.061 to 16.802 while FalseNeg dAP decreased from 32.172 to 28.215: more objects were found, but more erroneous candidates were introduced. | 0.65 |
| visibility/occlusion | Prior manual review observed occluded and difficult labels, but no Exp4 manual sample was collected. | 0.30 |
| assignment/NMS | Dupe dAP is only 0.667; this does not support NMS as the primary limiter. | 0.15 |
| experiment implementation confound | The run used batch 2, no explicit pretrained-weight transfer, and a rebuilt P3-P5 path. These changes prevent attributing the whole outcome to the presence of P2 alone. | 0.95 |

## Top-3 failure modes
1. Wrong or ambiguous fine-grained class predictions, especially where tiny
   objects lack enough appearance information.
2. Small/occluded objects that remain difficult despite the finer P2 grid.
3. Additional background/class false positives introduced by the dense P2
   prediction surface.

## Interpretation

P2 helped the mechanism it was intended to help: AP-small and recall increased,
localization dAP decreased from 4.648 to 2.909, and FalseNeg dAP decreased. It
did not address the dominant classification limitation, and the extra dense
predictions raised FalsePos and classification impact. Therefore the correct
claim is that this P2 configuration has a negative net benefit, not that finer
features never help small objects.
