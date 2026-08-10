# exp_003_class_grouping — Hypothesis

## Question

Does grouping visually ambiguous VisDrone classes reduce classification failures
and make the detector more useful for a coarse three-class application task?

## Prediction (written before training)

Grouping `pedestrian+people -> person`,
`bicycle+motor+tricycle+awning-tricycle -> small_vehicle`, and
`car+van+truck+bus -> road_vehicle` should reduce class-confusion errors and
make per-group AP less uneven. This is a task reformulation, so its three-class
metrics must not be presented as a direct improvement over the original
ten-class mAP.

## Background reading / prior evidence

Exp1/Exp2 TIDE reports identify classification as the largest AP-impact error
category. Manual review also found several class boundaries ambiguous to a human
reviewer. Fully occluded or otherwise unverifiable ground truths are not edited
in this experiment; they are recorded as annotation uncertainty when observed.

## Baseline

`exp_001_baseline_640` is the procedural reference. Its ten-class metrics remain
the official benchmark baseline; Exp3 is evaluated as a new three-class task.

## Causal variable changed

The class taxonomy only. The original images and source labels remain unchanged;
the mapping CSV generates a remapped label view.

## Compensating implementation changes

None. Training uses the baseline 640px/100-epoch settings so resolution is not
changed at the same time as the taxonomy.
