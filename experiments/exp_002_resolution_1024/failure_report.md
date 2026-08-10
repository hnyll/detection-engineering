# exp_002_resolution_1024 — Failure Analysis

Protocol: FiftyOne review of ≥100 failures on the fixed-protocol val split.
Keep observable **outcomes**, observable **conditions**, and causal **hypotheses** separate.

## Outcomes (counts from FiftyOne evaluation / TIDE)
| Outcome | Count | Notes |
|---|---|---|
| TP |22793 | |
| FP (background) | 127037| |
| FN (miss) |15966 | |
| Duplicate | | |
| Wrong class | | |
| Poor localization | | |

## Conditions present in reviewed failures
| Condition | # failures where present | Notes |
|---|---|---|
| tiny (<32px) | | |
| crowded | | |
| occluded | | |
| blur | | |
| low-light | | |
| truncation | | |

## Root-cause hypotheses
| Hypothesis | Supporting evidence (views/crops/TIDE deltas) | Confidence (0–1) |
|---|---|---:|
| assignment failure | Not established. No pre-assignment outputs or controlled crowded-scene analysis; post-NMS review cannot isolate the assigner. | 0.10 |

| feature resolution | AP-small = 0.08266 versus AP-large = 0.52818; Miss dAP = 8.715. However, 1024px training did not improve AP-small under the fixed 640px evaluation. | 0.45 |

| NMS suppression | TIDE Dupe dAP = 0.423, which is small; no raw duplicate count was collected. | 0.10 |

| label noise | Possible fully occluded or ambiguous annotations observed during review, but no systematic adjudication was performed. | 0.25 |

| domain shift | Blur, low-light, and occlusion observations may contribute, but the review was non-random and had no prevalence baseline. | 0.20 |

## Top-3 provisional failure modes

1. **Classification confusion.** TIDE `Cls dAP = 22.331`, the largest individual error impact. This suggests class mistakes are the largest aggregate limiter, although the report has no raw class-confusion count. Evidence: `tide_report.json`, per-class AP in `metrics.json`.

2. **Missed objects, especially small objects.** TIDE `Miss dAP = 8.715`, `FalseNeg = 33.816`, and AP-small is `0.08266` versus AP-large `0.52818`. This indicates a substantial small-object/recall problem. Evidence: `metrics.json`, `tide_report.json`.

3. **Localization errors.** TIDE `Loc dAP = 4.598` and `Both dAP = 0.612`. This indicates box placement or size errors remain, although they are less impactful than classification and misses. Evidence: `tide_report.json`.