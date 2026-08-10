# exp_009_small_object_signal_audit — Hypothesis

## Question

Do native VisDrone small-object crops contain enough human-identifiable class
evidence to justify further model work, and how much of that evidence survives
full-image resizing to 640 and 960?

## Prediction (written before the audit)

At least part of the small-object population contains usable semantic signal at
native resolution. Simulated 960 crops should preserve more of it than
simulated 640 crops, especially for objects whose native shortest side is below
16 pixels.

This audit uses one GT-blinded human reviewer and side-by-side resolution
variants. Its thresholds are engineering evidence, not statistical confidence
bounds or a model-training result.

Evidence for **usable native signal** requires:

- at least 15 reviewed samples per class;
- native-crop macro accuracy of at least **0.50** across the six balanced
  classes; and
- native non-uncertain coverage of at least **0.70**.

Evidence for **resolution information loss** requires both:

- native accuracy exceeds simulated-640 accuracy by at least **0.15**; and
- simulated-960 accuracy exceeds simulated-640 accuracy by at least **0.08**.

Supporting diagnostics:

- accuracy, uncertainty, and coverage by native shortest-side bin;
- per-class confusion, especially car/truck/bus and bicycle/motorcycle/person;
- native minus simulated-960 accuracy, indicating remaining potential for
  native-resolution tiles;
- notes marking objects that are invisible, annotation-ambiguous, truncated,
  or identifiable only from context.

## Background evidence

Exp7 already provides controlled evidence that some useful signal exists: the
exact Exp6 checkpoint gained 0.06145 AP-small when inference resolution changed
from 640 to 960. Exp8 matched training to 960 but added only 0.00188 AP-small,
suggesting that simply training the full image at 960 no longer extracts much
additional small-object benefit.

Small objects dominate the custom six-class validation set: 26,586 of 38,759
boxes (68.59%) have COCO area below `32^2`. Their native median dimensions are
approximately 11 x 23 pixels for person and 16-21 pixels per side for the
vehicle classes. These boxes are limited but are not uniformly one- or
two-pixel blobs.

## Sampling protocol

The audit deterministically samples 20 COCO-small boxes per class (120 total),
balanced as evenly as availability permits across native shortest-side bins:

- `micro`: shortest side below 8 px;
- `tiny`: shortest side from 8 px to below 16 px;
- `small`: shortest side at least 16 px while area remains below `32^2`.

Balanced sampling makes class and size comparisons easier but deliberately
does not reproduce their natural prevalence. Samples are blinded to the GT
class during review. The answer key remains separate until review is complete.

## Rendered variants

Each target uses the same square field of view with four times the target's
longest side, clamped to a 64-256 pixel native context window. The target is
marked without displaying its class. Three views are rendered:

1. the crop after simulated full-image resize/letterbox to 640;
2. the crop after simulated full-image resize/letterbox to 960;
3. the native source-image crop.

Display upscaling cannot create new evidence; it only makes the source pixels
inspectable. Context is retained because aerial classes may depend on road
position, neighboring objects, shape, and scale. A tight-box-only test would
measure a different and unnecessarily harsh information ceiling.

## Interpretation boundaries

- Human recognizability is evidence about available information, not proof that
  YOLO can learn it.
- GT-blinded triptychs are efficient but not independently blinded across
  resolution variants; seeing the native view can influence lower-resolution
  judgments.
- A wrong answer may reflect source ambiguity or an incorrect/ambiguous GT
  label. Review notes must preserve that distinction.
- High crop accuracy bypasses object search and localization. If crops are
  recognizable while detector AP-small remains low, the next bottleneck is
  detection/localization/candidate handling rather than source semantics.
- Low native accuracy is stronger evidence of an information or taxonomy
  ceiling, especially when accompanied by high `uncertain`/`not_visible` use.
- A trained oracle crop classifier or multiple independent reviewers would be
  required for a stronger signal-ceiling claim.
