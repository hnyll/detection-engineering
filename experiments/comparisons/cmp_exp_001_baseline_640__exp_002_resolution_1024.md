# Comparison: exp_001_baseline_640 vs exp_002_resolution_1024

Protocol: `a9b8207b3cc7a070` | dataset: visdrone/val | metric: map50_95

| exp | seeds | mAP50-95 (±std) | AP_small | ms/img | FPS | params(M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| exp_001_baseline_640 | 3 | 0.17049 ±0.00074 | 0.08731 | 23.32667 | 50.13333 | 2.584 | 6.32 |
| exp_002_resolution_1024 | 3 | 0.1729 ±0.00188 | 0.08061 | 14.63667 | 75.4 | 2.584 | 6.32 |

## Verdicts

- **exp_002_resolution_1024** vs exp_001_baseline_640: Δmap50_95=0.00241 → significant improvement
