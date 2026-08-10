# exp_001_baseline_640 — Failure Analysis

## Scope and sampling

This report uses the seed-17 predictions and FiftyOne evaluation:

- Dataset: `visdrone-val-7ada432f`
- Evaluation field: `pred_exp_001_baseline_640_s17_20da4576`
- Reviewed failures: 48 total, approximately 25 FP and 25 FN labels
- Protocol confidence floor: `0.001`

The project default is 100 reviewed failures. This pass intentionally used 48 as an exploratory sample. The exported artifact contains 52 `reviewed` label tags, while `reviewed_failures` is 48; four reviewed tags were applied while cross-referencing non-FP/FN labels. Condition tags overlap and are not estimates of dataset-wide prevalence.

## Outcomes (FiftyOne counts and TIDE impact)

| Outcome | Count / impact | Notes |
|---|---:|---|
| TP | 23,131 | FiftyOne seed-17 count |
| FP (all unmatched predictions) | 128,910 | Not background-only; predictions were retained from `conf=0.001`, so this raw count is threshold-dependent |
| FN (all unmatched ground truth) | 15,628 | FiftyOne seed-17 count |
| Duplicate | N/A | TIDE Dupe dAP = 0.392; dAP is AP impact, not a box count |
| Wrong class | N/A | TIDE Cls dAP = 23.065; one manually reviewed box was tagged `wrong class` |
| Poor localization | N/A | TIDE Loc dAP = 4.648; Cls+Loc Both dAP = 0.656 |

The TIDE report has no raw error counts (`counts: null`). Its dAP values answer “how much AP would be recovered by oracle-fixing this error type,” not “how many boxes have this error.”

## Conditions present in the reviewed labels

| Condition | Exported tag count | Interpretation |
|---|---:|---|
| Tiny (FiftyOne normalized-area proxy) | 35 | Most frequently observed condition in this sample; not a dataset-wide percentage |
| Crowded | 0 | Not observed in the reviewed sample; does not prove crowded scenes are harmless |
| Occluded | 10 | Repeated visibility issue, with possible overlap with other tags |
| Blur | 6 | Observed in a minority of reviewed labels |
| Low-light | 11 | Observed repeatedly, but no prevalence denominator was collected |
| Truncation | 0 | Not observed in this sample |

The `tiny_gt` view is a normalized-area screening proxy, not a literal guarantee that both box dimensions are below 32 pixels. The condition counts are descriptive observations from this review sample, not causal estimates.

## Root-cause hypotheses

| Hypothesis | Supporting evidence | Confidence |
|---|---|---:|
| Feature resolution | 35 exported `tiny` tags; seed-17 AP-small is 0.08616 versus AP-large 0.44568. The aggregate three-seed values are 0.08731 versus 0.41504. This supports a size-related failure pattern, but does not prove resolution is causal. | 0.65 |
| Classification confusion | TIDE Cls dAP = 23.065, the largest individual error impact. Only one reviewed label was tagged `wrong class`, so the specific class pairs remain weakly characterized. | 0.35 |
| Domain/visibility conditions | 10 occluded, 11 low-light, and 6 blur tags were recorded. Tags may overlap, and there is no matched base-rate sample. | 0.30 |
| Assignment failure | No crowded tags were recorded, and visual misses cannot isolate the training assigner from resolution, occlusion, or capacity. | 0.10 |
| NMS suppression | No duplicate tags were recorded and TIDE Dupe dAP is small. Post-NMS predictions cannot prove that a correct box was suppressed. | 0.10 |
| Label noise | One wrong-class observation is not enough to establish annotation noise. | 0.10 |

These confidence values are provisional judgments for prioritization, not probabilities and not causal proof.

## Top-3 provisional failure modes

1. **Small-object failures.** Tiny objects were the most frequently tagged condition in the reviewed sample, consistent with the large AP-small versus AP-large gap. Evidence: `fiftyone_review.json`, seed-17 review; `metrics.json`, AP-small/AP-large.

2. **Visibility-related failures.** Occlusion, low-light, and blur appeared repeatedly, but their relative importance is uncertain because the sample was not designed to estimate condition prevalence. Evidence: `fiftyone_review.json` label tags.

3. **Classification remains the largest aggregate error impact, but its visual mechanism is unresolved.** TIDE Cls dAP = 23.065 versus Miss dAP = 8.359, while only one reviewed label was explicitly tagged wrong-class. A larger, class-pair-focused review is needed before claiming that a specific confusion pair or label problem is dominant. Evidence: `tide_report.json`, `fiftyone_review.json`.

Background false positives and duplicate predictions should be treated as measured outcomes or follow-up questions, not ranked root causes from this sample.
