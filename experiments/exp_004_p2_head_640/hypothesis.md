# exp_004_p2_head_640 — Hypothesis

## Question

Does adding a P2/stride-4 detection head improve small-object recall on the
original ten-class VisDrone Task 1 without an unacceptable speed or memory cost?

## Prediction (written before training)

The P2 head should improve AP-small by at least 0.015 and mAP50-95 by at least
0.010 relative to `exp_001_baseline_640`, while reducing TIDE Miss dAP. The
tradeoff is higher memory use, lower throughput, and a batch-size reduction.

## Background reading / prior evidence

Exp1 and Exp2 have a large AP-small/AP-large gap and substantial TIDE Miss dAP.
Exp2's 1024px training intervention did not meet its AP-small or mAP success
threshold under the fixed 640px evaluation, so this experiment tests a different
small-object mechanism while keeping the official ten-class taxonomy.

## Baseline

`exp_001_baseline_640` (YOLO11n, ten classes, 640px, seed 17).

## Causal variable changed

The detection architecture: add one P2/stride-4 output branch. Dataset, classes,
evaluation protocol, seed, image size, and epoch budget remain unchanged.

## Compensating implementation changes

The planned batch-size reduction was 16 to 8, but the completed run required
batch 2 for GPU memory safety. This is a substantial hardware compensation and
must be treated as a limitation of the ablation.

## Result (written after evaluation)

The prediction was not confirmed. Against Exp1 seed 17, Exp4 improved AP-small
from 0.08616 to 0.09465 (+0.00849) and AR from 0.25307 to 0.26817, but reduced
mAP50-95 from 0.17026 to 0.16470. AP-medium fell from 0.26272 to 0.24558 and
AP-large from 0.44568 to 0.34997. Neither preregistered improvement threshold
was met.

The direction of the size-specific result is consistent with the P2 mechanism:
the finer feature map found more small objects. The net result was negative
because the recall gain came with more false-positive/classification impact and
worse medium/large performance.

## Validity limitation discovered after training

This run is not a clean P2-only causal test. The custom YAML rebuilt the
bottom-up P3-P5 path from the new P2 feature rather than preserving the stock
P3-P5 path. It was also instantiated from YAML and therefore did not explicitly
transfer `yolo11n.pt` pretrained weights, despite the generic
`pretrained: true` value in Ultralytics' saved arguments. The result supports
rejecting this configuration, but it does not prove that a pretrained,
append-only P2 head is ineffective.
