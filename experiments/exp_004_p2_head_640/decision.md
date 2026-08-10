---
exp: "exp_004_p2_head_640"
hypothesis_confirmed: false
outcome: negative
next_experiment: "exp_005_class_balanced_sampling"
---

# exp_004_p2_head_640 — Decision

## Problem

The ten-class Task 1 baseline has a large small-object gap and substantial
missed-object impact. Increasing training resolution to 1024px did not produce
the predicted improvement under the fixed 640px evaluation.

## Symptoms (metrics)

- Against Exp1 seed 17, AP-small improved from **0.08616** to **0.09465**
  (+0.00849) and AR improved from **0.25307** to **0.26817** (+0.01510).
- Overall mAP50-95 declined from **0.17026** to **0.16470**. AP-medium declined
  from **0.26272** to **0.24558**, and AP-large declined from **0.44568** to
  **0.34997**.
- TIDE Cls dAP increased from **23.065** to **24.866** and FalsePos dAP from
  **14.061** to **16.802**. Miss dAP decreased slightly from **8.359** to
  **8.016**, Loc from **4.648** to **2.909**, and FalseNeg from **32.172** to
  **28.215**.
- Exp4 uses **2.660M parameters** and **10.21 GFLOPs**, versus Exp1's 2.584M
  parameters and 6.32 GFLOPs. It adds about 61.6% theoretical compute for about
  2.9% more parameters.
- The completed run used batch **2**, not the planned batch 8. Its best internal
  validation mAP50-95 occurred at epoch 90, so the negative result is not
  explained by stopping at the apparent epoch-40 plateau.
- The preregistered thresholds (+0.015 AP-small and +0.010 mAP50-95) were not
  met. Exp4 received only one seed because the screening result was negative.

## Possible causes (ranked)

1. **Classification and taxonomy ambiguity.** Cls dAP 24.866 is the dominant
   individual TIDE category. A finer grid preserves spatial detail but does not
   create semantic evidence for distinguishing visually similar classes. The
   training labels are also strongly imbalanced: car has **144,866** boxes and
   pedestrian **79,337**, while awning-tricycle has **3,246**, tricycle
   **4,812**, bus **5,926**, and bicycle **10,480**. This makes imbalance a
   measured candidate cause, although it does not prove a particular confusion
   pair.

2. **Insufficient source information.** At the native 960x540 resolution, many
   VisDrone objects contain only a few pixels. P2 reduces feature-map
   downsampling but cannot recover texture or shape that is absent in the
   source image.

3. **Dense, shallow P2 predictions.** At 640px, P2 adds a 160x160 prediction
   surface. Total prediction locations increase from 8,400 for P3-P5 to 34,000
   for P2-P5. These shallow features carry fine position information but weaker
   semantics, creating more background and wrong-class opportunities.

4. **Implementation confounding.** The custom model did not explicitly load
   `yolo11n.pt` pretrained weights and rebuilt the P3-P5 bottom-up path instead
   of preserving the stock path. This is not a clean append-only P2 ablation.

5. **Batch-size compensation.** Batch 2 changes per-step BatchNorm statistics
   and optimization behavior relative to the batch-16 baseline, even though
   Ultralytics uses nominal-batch gradient accumulation.

## Evidence from previous experiments

Exp1 established the small-object gap but also showed that Cls dAP (23.065) was
much larger than Miss (8.359) or Loc (4.648). Exp2 trained at 1024px but did not
improve AP-small under the fixed 640px evaluation, providing evidence that
training resolution alone is not the primary solution. Exp4 improved small
object AP and recall while worsening classification/false-positive impact,
which reinforces that feature resolution is only part of the problem.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Architecture | Current P2/stride-4 implementation | Reject; negative net benefit and confounded implementation |
| Data diagnosis | Measure class frequencies, confusion pairs, and a targeted wrong-class review | Select before choosing a classification intervention |
| Data sampling | Balanced sampling or realistic targeted copy-paste for confirmed weak classes | Candidate; test as an isolated change after diagnosis |
| Augmentation | Stronger generic mosaic/blur/color augmentation | Do not select by default; tiny objects already have limited class cues |
| Model capacity | Pretrained YOLO11s or another stronger semantic backbone at 640px | Candidate classification-focused architecture ablation |
| Loss | Class weighting or focal-style classification loss | Defer until imbalance/hard-negative evidence is measured |
| Input/inference | Native-resolution or tiled inference | Separate protocol experiment; may preserve pixels but changes latency/comparability |
| Annotation | Manual GT correction | Defer; changes the benchmark labels |
| NMS/assignment | NMS or assigner changes | Defer; duplicate impact is small |

## Tradeoffs

Class-balanced sampling is inexpensive and preserves the official taxonomy, but
can overfit rare examples or damage calibration if frequency is not the actual
cause. Targeted copy-paste can add rare objects but may introduce unrealistic
context or edge artifacts. A larger pretrained backbone should provide stronger
semantic features, but costs VRAM, training time, and inference latency.

Generic stronger augmentation is not automatically helpful. The completed run
already used mosaic, HSV jitter, scale, translation, and horizontal flips.
Heavy blur, aggressive cropping, or more mosaic can remove the few visual cues
available for tiny classes. Use condition-specific augmentation only when the
review shows a corresponding failure mode.

The source images are 960x540. A 640px full-image input reduces their content to
approximately 640x360 before padding, so evaluating near 960px could preserve
native pixels. Scaling beyond native resolution creates no new information, and
Exp2 showed that 1024px training followed by 640px evaluation does not provide a
meaningful benefit. Native-resolution or tiled inference must be documented as
a separate evaluation protocol.

## Experiments to validate

1. Produce a class-frequency table and normalized confusion matrix, then review
   25-50 high-confidence wrong-class examples focused on car/van/truck/bus,
   pedestrian/people, and bicycle/tricycle/motor/awning-tricycle. This separates
   imbalance from intrinsically ambiguous labels.
2. The weak classes are underrepresented, so run one class-balanced image
   sampling or targeted oversampling ablation using the pretrained stock model.
   Keep augmentation, validation labels, and the official ten-class taxonomy
   unchanged so class balance is the only causal change.
3. If errors persist despite balanced data, test a pretrained YOLO11s at 640px
   as a semantic-capacity ablation. Compare per-class AP and Cls dAP, not only
   overall mAP.
4. Treat native-resolution or tiled inference as a separate deployment/protocol
   experiment. Do not mix it into the fixed-640 training ablation.
5. A corrected P2 experiment is optional: preserve stock P3-P5, append only P2,
   and explicitly transfer compatible `yolo11n.pt` weights. It is lower priority
   because TIDE says classification, not localization, is the larger limiter.

## Recommendation

Do not adopt Exp4's P2 model and do not spend two more seeds confirming it. It
improved small-object recall but failed both success thresholds, reduced overall
and medium/large AP, and increased classification/false-positive impact.

Keep Exp2 as the current ten-class candidate. Diagnose the actual class-pair and
frequency failures before changing augmentation or loss. The next modeling
ablation should target semantic classification capacity or measured class
imbalance. Do not claim that P2 generally fails; claim only that this
configuration produced a negative net result and was not a clean causal test.
