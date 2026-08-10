# exp_006_coco_aligned_balanced — Hypothesis

## Question

Does combining ambiguous VisDrone labels into six COCO-like concepts, while
retaining repeat-factor sampling, produce a more learnable application taxonomy?

## Prediction (written before training)

The custom six-class task should reduce TIDE classification impact and improve
the weakest grouped-class AP relative to a ten-class model because it removes
the pedestrian/people, car/van, and motor/tricycle/awning-tricycle boundaries.
The result is useful only if no grouped class collapses to near-zero AP and the
taxonomy matches the intended application.

Raw mAP is not directly comparable to Exp1, Exp2, Exp4, or Exp5 because Exp6
changes the task from ten classes to six. An increase in mAP cannot be described
as an improvement on official VisDrone Task 1.

## Proposed mapping

| VisDrone source | Exp6 target | Alignment quality |
|---|---|---|
| pedestrian, people | person | Strong; COCO has one person class |
| bicycle | bicycle | Direct |
| car, van | car | Approximate; COCO has no separate van class |
| truck | truck | Direct |
| bus | bus | Direct |
| motor, tricycle, awning-tricycle | motorcycle | Approximate; COCO has no three-wheel classes |

## Background evidence

Exp1-Exp4 consistently identify classification as the largest TIDE error type.
Manual review found the merged source boundaries ambiguous, and the rare source
classes are strongly imbalanced. Exp5 tests balancing while preserving the
official task; Exp6 intentionally combines that sampling policy with a coarser
application taxonomy.

Matching COCO names does not directly reuse the six COCO classifier logits:
Ultralytics replaces the incompatible 80-class output layer during fine-tuning.
The benefit, if any, comes from a simpler taxonomy aligned with concepts already
represented by the pretrained backbone, not from copying final class weights.

## Baseline and interpretation

Exp5 is the workflow parent because repeat-factor sampling is retained, but its
ten-class metrics are not a numerical control for this six-class task. This is a
composite application experiment, not a one-variable benchmark ablation.

## Causal changes

1. Remap ten labels into six COCO-like groups for train, validation, and test.
2. Recompute repeat-factor sampling from the six-class training labels.

Model, pretrained checkpoint, input size, augmentation policy, epoch budget,
batch size, and seed remain unchanged from Exp5.

The generated seed-17 manifest contains 8,716 entries from 6,471 source images
(1.35x). The class-level repeat factors are 1.60 for bus, 1.37 for bicycle,
1.21 for truck, 1.07 for motorcycle, and 1.00 for person and car. Because the
sampling unit is an image, common classes that co-occur with a repeated class
are repeated too; this is targeted oversampling, not exact instance balance.

## Guardrails

- Keep Exp5 as the official ten-class result.
- Never compare Exp6 raw mAP to ten-class mAP as if the tasks were identical.
- Report the mapping with every Exp6 metric.
- Treat van-to-car and three-wheel-to-motorcycle as application assumptions.
