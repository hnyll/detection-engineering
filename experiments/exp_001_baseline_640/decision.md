---
exp: "exp_001_baseline_640"
hypothesis_confirmed: false
outcome: inconclusive
next_experiment: "exp_002_resolution_1024"
---

# exp_001_baseline_640 — Decision

## Problem

The stock YOLO11n 640px baseline underperformed the preregistered overall accuracy range and showed a large small-object gap. TIDE identified classification as the largest individual AP-impact category, but the 48-box manual review did not contain enough wrong-class examples to identify the responsible class pairs. The next experiment should first test the strongest supported mechanism—limited information for small objects—without claiming that classification confusion has been explained.

## Symptoms (seed 17 and aggregate metrics)

- Seed-17 mAP50-95 is **0.17026** and mAP50 is **0.29306**, below the preregistered ranges of 0.18–0.22 and 0.32–0.38.
- Seed-17 AP-small is **0.08616** versus AP-large **0.44568**. The three-seed aggregate is AP-small **0.08731** versus AP-large **0.41504**.
- Seed-17 car AP50-95 is **0.49711**, while bicycle is **0.02953** and awning-tricycle is **0.04319**.
- Seed-17 TIDE dAP is Cls **23.065**, Miss **8.359**, Loc **4.648**, Bkg **1.750**, Both **0.656**, and Dupe **0.392**. Cls is the largest individual error impact; these are not raw error counts.
- FiftyOne reviewed 48 FP/FN labels. The exported tags included tiny 35, occluded 10, low-light 11, blur 6, and wrong-class 1. This is an exploratory sample below the project’s default 100-label review target.
- Seed-17 latency was noisy: 32.77 ms mean, 23.31 ms p50, and 65.32 ms p95 on the RTX 4060. It is not used to select the next training intervention.

## Possible causes, ranked

1. **Feature/resolution limitation — strongest provisional explanation.** The AP-small/AP-large gap and the frequency of tiny tags point toward insufficient visual information at 640px. This is an association, not proof; the resolution ablation is the test.

2. **Classification confusion — important unresolved explanation.** TIDE gives Cls the largest AP impact, but the manual sample contains only one explicit wrong-class tag. The report does not yet justify a specific pair such as car/van or truck/car.

3. **Visibility conditions — plausible secondary explanation.** Occlusion, low-light, and blur were observed, but no matched base-rate sample was collected, so their relative contribution is uncertain.

Assignment failure, NMS suppression, and label noise remain low-confidence hypotheses. The current post-NMS review cannot establish them.

## Evidence from previous experiments

N/A — `exp_001` is the first VisDrone modeling experiment. `exp_000_phase0_smoke` verified pipeline mechanics on COCO128 and provides no modeling evidence.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Input/data | Train at 1024px; compensate batch size for VRAM | Selected as the cleanest test of the strongest supported hypothesis |
| Data sampling | Class-balanced sampling or targeted copy-paste for specific confusion pairs | Defer until a larger review identifies repeated class pairs |
| Loss | Class weighting or focal-style classification changes | Defer; current evidence does not isolate imbalance as the cause |
| Assignment/NMS | Assigner or NMS changes | Defer; neither is supported by the current review |
| Architecture | Different neck/backbone | Defer; too many variables and no evidence requiring it |
| Training schedule | Longer schedule or different LR | Defer; not enough evidence that under-training is the primary issue |

## Tradeoffs

The 1024px experiment increases VRAM use, training time, and inference cost. Batch size will need to be recorded as a compensating implementation change. It may improve small-object recall without fixing class confusion, so the follow-up must measure AP-small, per-class AP, TIDE Cls/Miss dAP, and car AP rather than relying only on overall mAP. A class-balanced data intervention would target confusion more directly but has higher data-engineering cost and is not yet supported by enough reviewed examples.

## Experiment to validate

Create `exp_002_resolution_1024` with one causal change: training `imgsz: 1024`. Keep the model, dataset, epoch budget, augmentation policy, evaluation dataset, and seed set unchanged. Record any batch-size reduction separately as a hardware compensation. Keep the evaluation protocol explicit and unchanged for comparability; if evaluating at 1024 instead of the fixed 640px protocol, treat that as a protocol change and document it separately.

Initial falsifiable prediction:

- AP-small improves by at least **0.015**;
- overall mAP50-95 improves by at least **0.010**;
- car AP50-95 does not fall by more than **0.005**;
- TIDE Miss dAP decreases or remains stable.

Screen with seed 17 first. If the result is promising, confirm with seeds 42 and 1337 under the same protocol.

## Recommendation

Proceed with `exp_002_resolution_1024` as a provisional, one-variable ablation. The baseline establishes a reproducible low-resolution reference and provides a strong small-object signal, but the manual review is below the default sample size and does not yet justify a class-specific or NMS intervention. Revisit classification confusion after the 1024px result or after a larger class-pair-focused review.
