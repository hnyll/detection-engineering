# Threshold sweep: exp_012_hybrid_fullframe_tiles_640 seed 17

> This sweep only calibrates confidence thresholds on a frozen, post-NMS, post-max_det prediction artifact. It is not a model improvement, does not recover suppressed detections, and is tuned on this labeled split.

## Provenance and matching

- Dataset: `visdrone_exp006_coco_aligned/val`; GT fingerprint `e1142868d4dc2e9d`
- Protocol: `051e7ff3d9dcb3e0`; checkpoint `d078cf0e25577202`
- Predictions: `predictions_val_s17.json` (`a532e51fa6247ab0`, 163,908 boxes)
- Matching: class-aware, prediction-score-descending greedy one-to-one at IoU >= 0.50
- Threshold inclusion: `score >= threshold`; equal-score ties retain artifact order

## Overall sweep

| conf | micro P | micro R | micro F1 | macro P | macro R | macro F1 | FP/image | unmatched GT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.2612 | 0.8379 | 0.3983 | 0.2364 | 0.7133 | 0.3512 | 167.61 | 6,282 |
| 0.10 | 0.3199 | 0.8237 | 0.4608 | 0.2868 | 0.6962 | 0.4035 | 123.86 | 6,834 |
| 0.15 | 0.3816 | 0.8044 | 0.5176 | 0.3383 | 0.6768 | 0.4493 | 92.21 | 7,580 |
| 0.20 | 0.4472 | 0.7819 | 0.5690 | 0.3872 | 0.6532 | 0.4854 | 68.35 | 8,455 |
| 0.25 | 0.5123 | 0.7570 | 0.6111 | 0.4383 | 0.6279 | 0.5155 | 50.96 | 9,419 |
| 0.30 | 0.5734 | 0.7291 | 0.6419 | 0.4851 | 0.6011 | 0.5353 | 38.37 | 10,499 |
| 0.35 | 0.6316 | 0.6999 | 0.6640 | 0.5348 | 0.5734 | 0.5494 | 28.87 | 11,632 |
| 0.40 | 0.6854 | 0.6685 | 0.6768 | 0.5851 | 0.5448 | 0.5567 | 21.71 | 12,848 |
| 0.50 | 0.7698 | 0.5898 | 0.6679 | 0.6711 | 0.4712 | 0.5376 | 12.47 | 15,899 |

## Selected thresholds

Recommended global threshold from this candidate grid: **0.40** (maximum macro F1 0.5567; exact ties choose the higher threshold).

Micro-F1 optimum: **0.40** (0.6768).

| class | best conf | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 0.40 | 0.6860 | 0.5627 | 0.6183 | 7,861 | 3,599 | 6,108 |
| bicycle | 0.35 | 0.4011 | 0.2836 | 0.3323 | 365 | 545 | 922 |
| car | 0.50 | 0.7849 | 0.8170 | 0.8006 | 13,104 | 3,592 | 2,935 |
| truck | 0.50 | 0.4923 | 0.3827 | 0.4306 | 287 | 296 | 463 |
| bus | 0.50 | 0.6682 | 0.5697 | 0.6151 | 143 | 71 | 108 |
| motorcycle | 0.40 | 0.6251 | 0.5333 | 0.5756 | 3,447 | 2,067 | 3,016 |

## Per-class results

### Confidence 0.05

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.2234 | 0.7877 | 0.3481 | 11,004 | 38,248 | 2,965 |
| bicycle | 0.1396 | 0.4716 | 0.2154 | 607 | 3,742 | 680 |
| car | 0.3555 | 0.9420 | 0.5162 | 15,109 | 27,391 | 930 |
| truck | 0.1563 | 0.5987 | 0.2479 | 449 | 2,424 | 301 |
| bus | 0.3366 | 0.6853 | 0.4514 | 172 | 339 | 79 |
| motorcycle | 0.2067 | 0.7947 | 0.3281 | 5,136 | 19,707 | 1,327 |

### Confidence 0.10

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.2758 | 0.7716 | 0.4064 | 10,779 | 28,301 | 3,190 |
| bicycle | 0.1780 | 0.4460 | 0.2544 | 574 | 2,651 | 713 |
| car | 0.4189 | 0.9347 | 0.5785 | 14,991 | 20,798 | 1,048 |
| truck | 0.2066 | 0.5733 | 0.3038 | 430 | 1,651 | 320 |
| bus | 0.3817 | 0.6813 | 0.4893 | 171 | 277 | 80 |
| motorcycle | 0.2597 | 0.7705 | 0.3885 | 4,980 | 14,197 | 1,483 |

### Confidence 0.15

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.3322 | 0.7475 | 0.4600 | 10,442 | 20,988 | 3,527 |
| bicycle | 0.2153 | 0.4180 | 0.2842 | 538 | 1,961 | 749 |
| car | 0.4819 | 0.9241 | 0.6334 | 14,821 | 15,935 | 1,218 |
| truck | 0.2529 | 0.5480 | 0.3461 | 411 | 1,214 | 339 |
| bus | 0.4275 | 0.6813 | 0.5253 | 171 | 229 | 80 |
| motorcycle | 0.3197 | 0.7421 | 0.4469 | 4,796 | 10,205 | 1,667 |

### Confidence 0.20

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.4002 | 0.7184 | 0.5140 | 10,036 | 15,044 | 3,933 |
| bicycle | 0.2578 | 0.3869 | 0.3094 | 498 | 1,434 | 789 |
| car | 0.5407 | 0.9138 | 0.6794 | 14,657 | 12,452 | 1,382 |
| truck | 0.2896 | 0.5267 | 0.3737 | 395 | 969 | 355 |
| bus | 0.4528 | 0.6693 | 0.5402 | 168 | 203 | 83 |
| motorcycle | 0.3822 | 0.7040 | 0.4954 | 4,550 | 7,356 | 1,913 |

### Confidence 0.25

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.4746 | 0.6849 | 0.5607 | 9,567 | 10,589 | 4,402 |
| bicycle | 0.2959 | 0.3481 | 0.3199 | 448 | 1,066 | 839 |
| car | 0.5927 | 0.9030 | 0.7156 | 14,484 | 9,955 | 1,555 |
| truck | 0.3363 | 0.5053 | 0.4038 | 379 | 748 | 371 |
| bus | 0.4868 | 0.6614 | 0.5608 | 166 | 175 | 85 |
| motorcycle | 0.4434 | 0.6647 | 0.5319 | 4,296 | 5,393 | 2,167 |

### Confidence 0.30

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.5461 | 0.6471 | 0.5923 | 9,039 | 7,512 | 4,930 |
| bicycle | 0.3425 | 0.3193 | 0.3305 | 411 | 789 | 876 |
| car | 0.6399 | 0.8911 | 0.7449 | 14,292 | 8,044 | 1,747 |
| truck | 0.3656 | 0.4733 | 0.4126 | 355 | 616 | 395 |
| bus | 0.5108 | 0.6574 | 0.5749 | 165 | 158 | 86 |
| motorcycle | 0.5056 | 0.6186 | 0.5564 | 3,998 | 3,909 | 2,465 |

### Confidence 0.35

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6166 | 0.6055 | 0.6110 | 8,458 | 5,259 | 5,511 |
| bicycle | 0.4011 | 0.2836 | 0.3323 | 365 | 545 | 922 |
| car | 0.6825 | 0.8769 | 0.7675 | 14,064 | 6,544 | 1,975 |
| truck | 0.3947 | 0.4547 | 0.4226 | 341 | 523 | 409 |
| bus | 0.5439 | 0.6414 | 0.5887 | 161 | 135 | 90 |
| motorcycle | 0.5703 | 0.5784 | 0.5743 | 3,738 | 2,816 | 2,725 |

### Confidence 0.40

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6860 | 0.5627 | 0.6183 | 7,861 | 3,599 | 6,108 |
| bicycle | 0.4620 | 0.2455 | 0.3206 | 316 | 368 | 971 |
| car | 0.7221 | 0.8606 | 0.7853 | 13,803 | 5,313 | 2,236 |
| truck | 0.4265 | 0.4333 | 0.4299 | 325 | 437 | 425 |
| bus | 0.5889 | 0.6335 | 0.6104 | 159 | 111 | 92 |
| motorcycle | 0.6251 | 0.5333 | 0.5756 | 3,447 | 2,067 | 3,016 |

### Confidence 0.50

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.8009 | 0.4487 | 0.5752 | 6,268 | 1,558 | 7,701 |
| bicycle | 0.5692 | 0.1694 | 0.2611 | 218 | 165 | 1,069 |
| car | 0.7849 | 0.8170 | 0.8006 | 13,104 | 3,592 | 2,935 |
| truck | 0.4923 | 0.3827 | 0.4306 | 287 | 296 | 463 |
| bus | 0.6682 | 0.5697 | 0.6151 | 143 | 71 | 108 |
| motorcycle | 0.7112 | 0.4394 | 0.5432 | 2,840 | 1,153 | 3,623 |

## Limitations

- The prediction artifact is already post-NMS and post-max_det, so thresholding cannot recover suppressed or truncated boxes.
- The selected threshold is tuned and measured on the same validation split; confirm it on untouched labeled data before deployment claims.
- F1 depends on the configured class-aware IoU criterion and is not COCO mAP or a replacement for the fixed-protocol metrics.
- The harness GT has no COCO crowd/ignore annotations; omitted VisDrone ignore regions can inflate apparent false positives.
- Per-class threshold selection has greater overfitting risk than one global threshold and requires held-out confirmation.
- Prediction scores are stored at finite precision in the evaluation artifact, so equality follows the serialized values.
