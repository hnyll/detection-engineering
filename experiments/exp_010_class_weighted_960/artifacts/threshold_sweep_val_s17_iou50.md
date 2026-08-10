# Threshold sweep: exp_010_class_weighted_960 seed 17

> This sweep only calibrates confidence thresholds on a frozen, post-NMS, post-max_det prediction artifact. It is not a model improvement, does not recover suppressed detections, and is tuned on this labeled split.

## Provenance and matching

- Dataset: `visdrone_exp006_coco_aligned/val`; GT fingerprint `e1142868d4dc2e9d`
- Protocol: `821306fe3a07ceac`; checkpoint `d078cf0e25577202`
- Predictions: `predictions_val_s17.json` (`310c355451e114e3`, 153,156 boxes)
- Matching: class-aware, prediction-score-descending greedy one-to-one at IoU >= 0.50
- Threshold inclusion: `score >= threshold`; equal-score ties retain artifact order

## Overall sweep

| conf | micro P | micro R | micro F1 | macro P | macro R | macro F1 | FP/image | unmatched GT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.3559 | 0.7430 | 0.4813 | 0.3454 | 0.5894 | 0.4253 | 95.11 | 9,960 |
| 0.10 | 0.4611 | 0.7137 | 0.5603 | 0.4317 | 0.5598 | 0.4797 | 58.99 | 11,095 |
| 0.15 | 0.5526 | 0.6831 | 0.6109 | 0.5054 | 0.5331 | 0.5126 | 39.12 | 12,282 |
| 0.20 | 0.6329 | 0.6526 | 0.6426 | 0.5663 | 0.5073 | 0.5295 | 26.78 | 13,463 |
| 0.25 | 0.7019 | 0.6215 | 0.6592 | 0.6293 | 0.4814 | 0.5383 | 18.67 | 14,671 |
| 0.30 | 0.7622 | 0.5902 | 0.6652 | 0.6785 | 0.4550 | 0.5355 | 13.02 | 15,885 |
| 0.35 | 0.8131 | 0.5579 | 0.6618 | 0.7272 | 0.4285 | 0.5264 | 9.07 | 17,134 |
| 0.40 | 0.8560 | 0.5236 | 0.6497 | 0.7659 | 0.4007 | 0.5098 | 6.23 | 18,466 |
| 0.50 | 0.9122 | 0.4510 | 0.6036 | 0.8302 | 0.3427 | 0.4617 | 3.07 | 21,277 |

## Selected thresholds

Recommended global threshold from this candidate grid: **0.25** (maximum macro F1 0.5383; exact ties choose the higher threshold).

Micro-F1 optimum: **0.30** (0.6652).

| class | best conf | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 0.25 | 0.6370 | 0.5045 | 0.5631 | 7,048 | 4,016 | 6,921 |
| bicycle | 0.15 | 0.2822 | 0.2331 | 0.2553 | 300 | 763 | 987 |
| car | 0.35 | 0.8738 | 0.7817 | 0.8252 | 12,537 | 1,810 | 3,502 |
| truck | 0.25 | 0.5949 | 0.3720 | 0.4578 | 279 | 190 | 471 |
| bus | 0.30 | 0.8000 | 0.5100 | 0.6229 | 128 | 32 | 123 |
| motorcycle | 0.25 | 0.5974 | 0.5016 | 0.5453 | 3,242 | 2,185 | 3,221 |

## Per-class results

### Confidence 0.05

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.2942 | 0.6577 | 0.4065 | 9,187 | 22,042 | 4,782 |
| bicycle | 0.1757 | 0.2984 | 0.2211 | 384 | 1,802 | 903 |
| car | 0.4832 | 0.8934 | 0.6272 | 14,330 | 15,325 | 1,709 |
| truck | 0.3030 | 0.4573 | 0.3645 | 343 | 789 | 407 |
| bus | 0.5480 | 0.5458 | 0.5469 | 137 | 113 | 114 |
| motorcycle | 0.2683 | 0.6836 | 0.3854 | 4,418 | 12,048 | 2,045 |

### Confidence 0.10

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.3849 | 0.6222 | 0.4756 | 8,691 | 13,887 | 5,278 |
| bicycle | 0.2275 | 0.2549 | 0.2404 | 328 | 1,114 | 959 |
| car | 0.5950 | 0.8773 | 0.7091 | 14,071 | 9,577 | 1,968 |
| truck | 0.3943 | 0.4253 | 0.4092 | 319 | 490 | 431 |
| bus | 0.6239 | 0.5418 | 0.5800 | 136 | 82 | 115 |
| motorcycle | 0.3646 | 0.6373 | 0.4638 | 4,119 | 7,179 | 2,344 |

### Confidence 0.15

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.4733 | 0.5831 | 0.5225 | 8,145 | 9,063 | 5,824 |
| bicycle | 0.2822 | 0.2331 | 0.2553 | 300 | 763 | 987 |
| car | 0.6770 | 0.8594 | 0.7574 | 13,784 | 6,576 | 2,255 |
| truck | 0.4679 | 0.3987 | 0.4305 | 299 | 340 | 451 |
| bus | 0.6802 | 0.5339 | 0.5982 | 134 | 63 | 117 |
| motorcycle | 0.4515 | 0.5903 | 0.5116 | 3,815 | 4,635 | 2,648 |

### Confidence 0.20

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.5564 | 0.5437 | 0.5500 | 7,595 | 6,056 | 6,374 |
| bicycle | 0.3312 | 0.2005 | 0.2498 | 258 | 521 | 1,029 |
| car | 0.7421 | 0.8412 | 0.7885 | 13,492 | 4,689 | 2,547 |
| truck | 0.5254 | 0.3867 | 0.4455 | 290 | 262 | 460 |
| bus | 0.7097 | 0.5259 | 0.6041 | 132 | 54 | 119 |
| motorcycle | 0.5329 | 0.5460 | 0.5394 | 3,529 | 3,093 | 2,934 |

### Confidence 0.25

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6370 | 0.5045 | 0.5631 | 7,048 | 4,016 | 6,921 |
| bicycle | 0.3857 | 0.1756 | 0.2413 | 226 | 360 | 1,061 |
| car | 0.7927 | 0.8207 | 0.8065 | 13,164 | 3,442 | 2,875 |
| truck | 0.5949 | 0.3720 | 0.4578 | 279 | 190 | 471 |
| bus | 0.7679 | 0.5139 | 0.6158 | 129 | 39 | 122 |
| motorcycle | 0.5974 | 0.5016 | 0.5453 | 3,242 | 2,185 | 3,221 |

### Confidence 0.30

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.7083 | 0.4633 | 0.5602 | 6,472 | 2,666 | 7,497 |
| bicycle | 0.4357 | 0.1500 | 0.2231 | 193 | 250 | 1,094 |
| car | 0.8371 | 0.8035 | 0.8200 | 12,887 | 2,507 | 3,152 |
| truck | 0.6329 | 0.3493 | 0.4502 | 262 | 152 | 488 |
| bus | 0.8000 | 0.5100 | 0.6229 | 128 | 32 | 123 |
| motorcycle | 0.6571 | 0.4537 | 0.5368 | 2,932 | 1,530 | 3,531 |

### Confidence 0.35

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.7699 | 0.4218 | 0.5450 | 5,892 | 1,761 | 8,077 |
| bicycle | 0.5375 | 0.1336 | 0.2141 | 172 | 148 | 1,115 |
| car | 0.8738 | 0.7817 | 0.8252 | 12,537 | 1,810 | 3,502 |
| truck | 0.6587 | 0.3293 | 0.4391 | 247 | 128 | 503 |
| bus | 0.8158 | 0.4940 | 0.6154 | 124 | 28 | 127 |
| motorcycle | 0.7075 | 0.4105 | 0.5195 | 2,653 | 1,097 | 3,810 |

### Confidence 0.40

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.8239 | 0.3807 | 0.5208 | 5,318 | 1,137 | 8,651 |
| bicycle | 0.5948 | 0.1072 | 0.1817 | 138 | 94 | 1,149 |
| car | 0.9036 | 0.7551 | 0.8227 | 12,111 | 1,292 | 3,928 |
| truck | 0.6735 | 0.3080 | 0.4227 | 231 | 112 | 519 |
| bus | 0.8414 | 0.4861 | 0.6162 | 122 | 23 | 129 |
| motorcycle | 0.7584 | 0.3672 | 0.4948 | 2,373 | 756 | 4,090 |

### Confidence 0.50

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.9041 | 0.2888 | 0.4377 | 4,034 | 428 | 9,935 |
| bicycle | 0.7388 | 0.0769 | 0.1393 | 99 | 35 | 1,188 |
| car | 0.9416 | 0.7027 | 0.8048 | 11,271 | 699 | 4,768 |
| truck | 0.7226 | 0.2640 | 0.3867 | 198 | 76 | 552 |
| bus | 0.8692 | 0.4502 | 0.5932 | 113 | 17 | 138 |
| motorcycle | 0.8050 | 0.2734 | 0.4082 | 1,767 | 428 | 4,696 |

## Limitations

- The prediction artifact is already post-NMS and post-max_det, so thresholding cannot recover suppressed or truncated boxes.
- The selected threshold is tuned and measured on the same validation split; confirm it on untouched labeled data before deployment claims.
- F1 depends on the configured class-aware IoU criterion and is not COCO mAP or a replacement for the fixed-protocol metrics.
- The harness GT has no COCO crowd/ignore annotations; omitted VisDrone ignore regions can inflate apparent false positives.
- Per-class threshold selection has greater overfitting risk than one global threshold and requires held-out confirmation.
- Prediction scores are stored at finite precision in the evaluation artifact, so equality follows the serialized values.
