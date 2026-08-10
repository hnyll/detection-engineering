# Comparison: exp_007_inference_960 vs exp_008_train_960

Protocol: `821306fe3a07ceac` | dataset: visdrone_exp006_coco_aligned/val | metric: map50_95

| exp | seeds | mAP50-95 (±std) | AP_small | ms/img | FPS | params(M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| exp_007_inference_960 | 1 | 0.26661 | 0.17284 | 10.77 | 92.9 | 2.583 | 14.22 |
| exp_008_train_960 | 1 | 0.28847 | 0.17472 | 10.4 | 96.1 | 2.583 | 14.22 |

## Verdicts

- **exp_008_train_960** vs exp_007_inference_960: Δmap50_95=0.02186 → INSUFFICIENT-SEEDS (significance needs ≥3 seeds per side; have 1 vs 1)
