---
exp: "exp_005_class_balanced_sampling"
hypothesis_confirmed: false
outcome: negative
next_experiment: "exp_006_coco_aligned_balanced"
---

# exp_005_class_balanced_sampling — Decision

## Problem

The official ten-class model's largest individual TIDE error is classification,
and several weak classes have far fewer source instances than car. Exp5 tested
whether repeat-factor image sampling could improve those classes without
damaging overall performance or the car guardrail.

## Symptoms and preregistered result

| Criterion | Exp1 s17 | Exp5 s17 | Delta | Decision threshold | Result |
|---|---:|---:|---:|---:|---|
| rare-class mean AP50-95 | 0.12661 | 0.12849 | +0.00189 | at least +0.015 | fail |
| overall mAP50-95 | 0.17026 | 0.17265 | +0.00239 | at least +0.005 | fail |
| TIDE Cls dAP | 23.065 | 23.250 | +0.185 (worse) | decrease by at least 1.5 | fail |
| car AP50-95 | 0.49711 | 0.49916 | +0.00205 | no decline greater than 0.010 | pass |

The sampler slightly improved bicycle (+0.00532) and bus (+0.00365), left
truck unchanged, and changed tricycle and awning-tricycle by less than 0.0004.
It improved AP-small by 0.00403 but did not improve mAP75. TIDE classification
remained the dominant individual error type, followed by misses and
localization.

The run completed normally. Trainer validation mAP50-95 peaked at epoch 83,
and `best.pt` was evaluated. The mild decline through epoch 100 was comparable
to Exp1, so overtraining is not a sufficient explanation for the failed
screen.

## Possible causes (ranked)

1. **The image-level sampler was too broad.** An epoch expanded from 6,471 to
   10,095 entries, but multi-class images repeated common objects along with
   rare ones. Target-class instances increased about 1.75x while car increased
   1.62x, improving aggregate rare-to-car exposure by only about 8.4%.

2. **Frequency is not the dominant classification limitation.** More exposure
   did not reduce Cls dAP or materially improve strict target-class AP. Tiny
   appearance, overlapping class definitions, and annotation ambiguity remain
   plausible, but Exp5 has no manual class-pair review to separate them.

3. **The source images provide limited evidence for tiny objects.** AP-small is
   0.09019, far below AP-medium 0.26691 and AP-large 0.41105. Repeating an image
   does not add pixels, viewpoints, or visual diversity.

4. **Repeated examples may give diminishing returns.** Validation plateaued
   around epochs 70-90, but the late behavior was not unusually worse than
   Exp1 and `best.pt` protected evaluation from the final-epoch decline.

5. **One-seed uncertainty remains.** The +0.00239 mAP change may or may not be
   repeatable. Additional seeds were conditional on passing the seed-17 screen,
   which this experiment did not do.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Data sampling | Current repeat-factor image sampler | Reject this configuration; insufficient targeted benefit for 1.56x entries per epoch |
| Data sampling | Class-aware batches or stronger targeted sampling | Possible official-task follow-up, but audit effective exposure ratios before training |
| Data augmentation | Targeted object copy-paste | Possible, but may introduce scale/context and boundary artifacts |
| Model capacity | Pretrained YOLO11s at 640px | Candidate semantic-capacity ablation; costs VRAM, latency, and training time |
| Loss | Class weighting or focal-style loss | Defer until confusion pairs and calibration effects are measured |
| Taxonomy | Six COCO-like classes plus balancing (Exp6) | Proceed only as a custom application task, not an official VisDrone improvement |
| NMS/assignment | NMS or assigner changes | Defer; duplicate impact is small and current evidence does not isolate assignment |

## Tradeoffs

Image repetition is simple and preserves the official annotations, but this
configuration adds 56% more entries and optimizer steps per epoch without
adding visual diversity. Because total training steps and exposure both change,
the small positive mAP delta cannot be attributed solely to better balance.

More aggressive class-aware sampling or loss weighting may prioritize rare
classes more effectively, but can overfit scarce examples, increase false
positives, and distort confidence calibration. Object copy-paste adds apparent
diversity but may create unrealistic context. A larger pretrained model may
improve semantic features but increases compute and deployment cost. Grouping
classes removes ambiguous distinctions but changes the task and permanently
discards fine-grained labels.

## Experiments to validate

1. Do not run Exp5 seeds 42 and 1337: the preregistered seed-17 screen failed.
2. Run Exp6 only as the planned custom six-class application experiment. Its
   metrics must not be compared directly with ten-class Exp1-Exp5 metrics.
3. If Exp6 appears useful, train an otherwise identical **unbalanced six-class
   control** before attributing any gain to balancing. Exp6 combines taxonomy
   and sampling and cannot isolate either intervention by itself.
4. Before another official ten-class balancing experiment, review 25-50
   high-confidence wrong-class cases and measure the actual confusion pairs.
   Then preregister a stronger class-aware exposure target or test a pretrained
   YOLO11s as a separate one-variable capacity ablation.

## Recommendation

Do not adopt Exp5's repeat-factor sampler as an improvement to the official
ten-class model, and do not spend two additional seeds confirming this exact
configuration. It passed the car guardrail but missed all three intended
benefit thresholds while increasing each epoch by 56%.

Record Exp5 as a useful negative result: class imbalance may still contribute,
but broad image repetition did not address the dominant classification error.
Proceed to Exp6 only for the custom COCO-like application taxonomy. Keep the
official ten-class track and its conclusions separate.
