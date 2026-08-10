# Comparison: exp_008_train_960 vs exp_010_class_weighted_960

Protocol: `821306fe3a07ceac` | dataset: visdrone_exp006_coco_aligned/val | metric: map50_95

| exp | seeds | mAP50-95 (±std) | AP_small | ms/img | FPS | params(M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| exp_008_train_960 | 1 | 0.28847 | 0.17472 | 10.4 | 96.1 | 2.583 | 14.22 |
| exp_010_class_weighted_960 | 1 | 0.29311 | 0.17903 | 10.55 | 94.7 | 2.583 | 14.22 |

## Verdicts

- **exp_010_class_weighted_960** vs exp_008_train_960: Δmap50_95=0.00464 → INSUFFICIENT-SEEDS (significance needs ≥3 seeds per side; have 1 vs 1)
