---
exp: "exp_002_resolution_1024"
hypothesis_confirmed: false
outcome: negative
next_experiment: "exp_004_p2_head_640"
---

# exp_002_resolution_1024 — Decision

## Problem
1. Classification confusion. classes become mixed/ambiguous. trucks, vehicles, cars look alike. pedestrians and people look alike. tricycle and motor look alike

2. Missed objects, especially small objects

3. Localization errors. boxes are not the greatest and could be better localized
## Symptoms (metrics)

Across three matched seeds, fixed-640 evaluation mAP50-95 increased from
`0.17049 ± 0.00074` to `0.17290 ± 0.00188` (`+0.00241`). The comparison
harness calls that difference statistically significant, but it is only a
`0.24` AP-point practical gain. AP-small moved in the wrong direction, from
`0.08731` to `0.08061` (`-0.00670`). Training at 1024 therefore did not support
the expected broad improvement or the small-object mechanism.

1. TIDE `Cls dAP = 22.331`, the largest individual error impact. This suggests class mistakes are the largest aggregate limiter, although the report has no raw class-confusion count. Evidence: seed-17 `tide_report.json`, per-class AP in `metrics.json`.

2. TIDE `Miss dAP = 8.715`, `FalseNeg = 33.816`, and AP-small is `0.08266` versus AP-large `0.52818`. This indicates a substantial small-object/recall problem. Evidence: `metrics.json`, `tide_report.json`.

3. TIDE `Loc dAP = 4.598` and `Both dAP = 0.612`. This indicates box placement or size errors remain, although they are less impactful than classification and misses. Evidence: `tide_report.json`.

## Possible causes (ranked)
1. Taxonomy and annotation ambiguity — moderate confidence (0.45).
   The TIDE `Cls dAP = 22.331` indicates that classification errors have the largest AP impact. Prior visual review found ambiguous boundaries such as car/van/truck/bus and pedestrian/people. However, Exp2 did not receive a systematic manual review, so this remains provisional rather than a measured class-pair diagnosis.

2. Insufficient information for small or occluded objects — moderate confidence (0.50).
   AP-small is `0.08266` versus AP-large `0.52818`, and TIDE `Miss dAP = 8.715`. This supports a small-object recall problem. However, training at 1024px did not improve AP-small under the fixed 640px evaluation, so resolution alone is not confirmed as the cause.

3. Box regression or annotation localization limits — low-to-moderate confidence (0.30).
   TIDE `Loc dAP = 4.598` and `Both dAP = 0.612` show a measurable localization contribution. Poor boxes may reflect model regression, tiny targets, or ambiguous/partially occluded ground truths; the current evidence cannot separate these causes.

4. NMS or assignment failure — low confidence (0.10).
   TIDE `Dupe dAP = 0.423` is small, and the post-NMS review does not expose assignment behavior or suppressed predictions. There is insufficient evidence to prioritize this cause.

5. Domain/visibility shift — low confidence (0.20).
   Blur, low-light, and occlusion were observed previously, but no systematic Exp2 sample or prevalence comparison was collected.

## Evidence from previous experiments

Exp1 established the three-seed ten-class control. Exp2 uses the same fixed
evaluation protocol and three seeds, so the aggregate comparison is valid.
Training resolution and the required batch reduction are the treatment and
compensating hardware change; inference resolution remains 640.

## Solution families considered
<!-- Data / Architecture / Loss / Assignment / Training / Deployment -->
lets target classification.

class balanced / focal loss

p2/high resolution detection head

tiled/sahi inference


## Tradeoffs

- **P2/high-resolution detection head:** Directly targets small-object misses and may improve AP-small and recall. It increases model complexity, VRAM use, training time, and inference cost, and could introduce more false positives.

- **Tiled/SAHI inference:** Gives small objects more pixels and may reduce misses. It increases latency and implementation complexity, can create tile-boundary artifacts, and changes the inference protocol, so it is not directly comparable to the fixed full-image benchmark.

- **Class-balanced sampling or loss:** May help rare classes and classification confusion. It can overfit rare examples, distort calibration, and may not solve inherently ambiguous labels.


## Experiments to validate

1. **P2/small-object architecture ablation.**
   Keep the original ten classes, 640px fixed evaluation, dataset, seed, and epoch budget unchanged. Change only the detection architecture to add a higher-resolution feature head. Success requires:
   - AP-small improvement of at least 0.015
   - mAP50–95 improvement of at least 0.010
   - lower or stable TIDE Miss dAP
   - no unacceptable latency increase

2. **Tiled inference evaluation.**
   Evaluate the best ten-class model with a documented tiling protocol. Treat this as a separate inference experiment because latency and the evaluation protocol change. Measure AP-small, recall, Miss dAP, and latency.

3. **Class-balanced training ablation.**
   If classification remains the dominant error after the small-object experiment, test class-balanced sampling or loss weighting as one isolated training change. Measure per-class AP, TIDE Cls dAP, and false-positive/false-negative behavior.

## Recommendation

Do not adopt 1024px training for the ten-class VisDrone Task 1. The three-seed
mAP gain is detectable but too small to justify the extra training cost, while
AP-small regressed. Retain Exp1 as the official fixed-640 reference and test a
single small-object intervention next. Exp3 is a separate coarse-taxonomy task
and must not replace the Task 1 result.
