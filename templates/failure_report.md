# {EXP} — Failure Analysis

Protocol: FiftyOne review of ≥100 failures on the fixed-protocol val split.
Keep observable **outcomes**, observable **conditions**, and causal **hypotheses** separate.

## Outcomes (counts from FiftyOne evaluation / TIDE)
| Outcome | Count | Notes |
|---|---|---|
| TP | | |
| FP (background) | | |
| FN (miss) | | |
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
|---|---|---|
| assignment failure | | |
| feature resolution | | |
| NMS suppression | | |
| label noise | | |
| domain shift | | |

## Top-3 failure modes (with evidence links)
1.
2.
3.
