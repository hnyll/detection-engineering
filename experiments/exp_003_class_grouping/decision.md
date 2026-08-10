---
exp: "exp_003_class_grouping"
hypothesis_confirmed: true
outcome: positive
next_experiment: "exp_006_coco_aligned_balanced"
---

# exp_003_class_grouping — Decision

## Problem

The original ten-class task contains class boundaries that are difficult to
resolve visually (for example car/van/truck/bus and pedestrian/people).

## Result

At seed 17, the coarse three-class task reached `0.27391` mAP50-95,
`0.51425` mAP50, and `0.18380` AP-small. TIDE classification opportunity fell
to `3.012` dAP; the dominant residual modes became localization (`10.959`) and
misses (`9.206`). This supports the hypothesis that collapsing ambiguous source
labels makes the coarse task easier to classify.

The result is not an improvement on official VisDrone Task 1. Both predictions
and ground truth use a new three-class taxonomy, and only one seed was run.

## Possible causes (ranked)

1. Taxonomy ambiguity creates penalties for plausible class choices.
2. Small/occluded objects remain difficult even after grouping.
3. Some apparent failures may be unverifiable or noisy ground truth.

## Evidence from previous experiments

Exp1 and Exp2 TIDE reports, plus the documented FiftyOne observations. Do not
claim that a grouping fixed annotation errors; this experiment changes the task.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Taxonomy/data | Three-group mapping in `class_groups.csv` | Selected for this experiment |
| Annotation QA | Manually correct suspicious boxes | Deferred; no manual relabeling in Exp3 |
| Input/resolution | 1024px training | Deferred; keep 640px fixed |

## Tradeoffs

Grouping may reduce class confusion but removes fine-grained information. It
also makes Exp3 metrics a different task, so the mapping and class definitions
must be reported with every result.

## Experiments to validate

If a deployment genuinely needs only three coarse groups, repeat seeds 42 and
1337 and complete a counted failure review. For this project, retain the
original ten-class branch for official VisDrone claims and use a separately
declared COCO-aligned six-class application taxonomy for later work.

## Recommendation

The grouping hypothesis is supported for the reformulated coarse task, but the
three-class model is not adopted because it no longer matches the requested
VisDrone Task 1 taxonomy. Preserve Exp3 as evidence that taxonomy was a major
classification constraint, not as a leaderboard result.
