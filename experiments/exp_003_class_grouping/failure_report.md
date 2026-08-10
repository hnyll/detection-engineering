# exp_003_class_grouping — Failure Analysis

This is a three-class taxonomy experiment. The source VisDrone labels were not
manually corrected, and no counted FiftyOne review was completed. The evidence
below therefore comes from fixed-protocol metrics and TIDE; blank manual counts
are not inferred from dAP.

## Outcomes (counts from FiftyOne evaluation / TIDE)
| Outcome | Count | Notes |
|---|---:|---|
| TP | | |
| FP (background) | | |
| FN (miss) | | |
| Duplicate | | |
| Wrong class | | |
| Poor localization | | |

TIDE impacts for seed 17 are `Cls 3.012`, `Loc 10.959`, `Both 0.623`,
`Dupe 0.922`, `Bkg 3.191`, and `Miss 9.206` dAP. These are independent oracle
AP opportunities, not counts and not additive.

## Conditions present in reviewed failures
| Condition | # failures where present | Notes |
|---|---:|---|
| tiny (<32px) | | |
| crowded | | |
| occluded | | |
| fully occluded / unverifiable GT | | |
| ambiguous source class | | |
| blur | | |
| low-light | | |

## Root-cause hypotheses
| Hypothesis | Supporting evidence (views/crops/TIDE deltas) | Confidence (0–1) |
|---|---|---:|
| taxonomy ambiguity | Grouping collapses the visually similar source labels and leaves only `3.012` Cls dAP, far below the original ten-class diagnostic. Because the GT taxonomy also changed, this is descriptive rather than a direct causal AP comparison. | 0.85 |
| feature/resolution limitation | AP-small is `0.18380` versus AP-medium `0.43009` and AP-large `0.60579`; Miss is `9.206` and Loc is `10.959` dAP. | 0.85 |
| label uncertainty | Source annotations were not adjudicated and no systematic manual review was completed. | 0.20 |
| NMS suppression | Dupe is only `0.922` dAP, well below Loc and Miss. | 0.10 |

## Top-3 failure modes (with evidence links)
1. **Localization (`10.959` dAP).** Coarse grouping removes many semantic
   boundaries, leaving box placement as the largest main TIDE opportunity.
2. **Missed objects (`9.206` dAP).** The large AP-small gap is consistent with
   tiny-object recall remaining difficult after taxonomy simplification.
3. **Background false positives (`3.191` dAP).** This is material but smaller
   than localization and misses; no manual prevalence claim is made.
