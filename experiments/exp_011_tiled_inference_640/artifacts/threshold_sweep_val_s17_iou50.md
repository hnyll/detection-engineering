# Threshold sweep: exp_011_tiled_inference_640 seed 17

> This sweep only calibrates confidence thresholds on a frozen, post-NMS, post-max_det prediction artifact. It is not a model improvement, does not recover suppressed detections, and is tuned on this labeled split.

## Provenance and matching

- Dataset: `visdrone_exp006_coco_aligned/val`; GT fingerprint `e1142868d4dc2e9d`
- Protocol: `feddb3567d439c3a`; checkpoint `d078cf0e25577202`
- Predictions: `predictions_val_s17.json` (`2e324ed84dc1325f`, 163,320 boxes)
- Matching: class-aware, prediction-score-descending greedy one-to-one at IoU >= 0.50
- Threshold inclusion: `score >= threshold`; equal-score ties retain artifact order

## Overall sweep

| conf | micro P | micro R | micro F1 | macro P | macro R | macro F1 | FP/image | unmatched GT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.2898 | 0.8274 | 0.4293 | 0.2565 | 0.6943 | 0.3705 | 143.41 | 6,688 |
| 0.10 | 0.3685 | 0.8073 | 0.5060 | 0.3215 | 0.6724 | 0.4325 | 97.86 | 7,468 |
| 0.15 | 0.4432 | 0.7834 | 0.5661 | 0.3808 | 0.6485 | 0.4787 | 69.62 | 8,397 |
| 0.20 | 0.5150 | 0.7579 | 0.6133 | 0.4340 | 0.6232 | 0.5111 | 50.48 | 9,383 |
| 0.25 | 0.5781 | 0.7322 | 0.6461 | 0.4831 | 0.5972 | 0.5330 | 37.80 | 10,381 |
| 0.30 | 0.6362 | 0.7039 | 0.6683 | 0.5284 | 0.5691 | 0.5451 | 28.47 | 11,475 |
| 0.35 | 0.6881 | 0.6741 | 0.6810 | 0.5746 | 0.5396 | 0.5509 | 21.62 | 12,631 |
| 0.40 | 0.7325 | 0.6419 | 0.6842 | 0.6213 | 0.5104 | 0.5513 | 16.58 | 13,878 |
| 0.50 | 0.8001 | 0.5634 | 0.6612 | 0.6971 | 0.4373 | 0.5199 | 9.95 | 16,924 |

## Selected thresholds

Recommended global threshold from this candidate grid: **0.40** (maximum macro F1 0.5513; exact ties choose the higher threshold).

Micro-F1 optimum: **0.40** (0.6842).

| class | best conf | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 0.35 | 0.6829 | 0.5758 | 0.6248 | 8,043 | 3,734 | 5,926 |
| bicycle | 0.30 | 0.3778 | 0.2859 | 0.3255 | 368 | 606 | 919 |
| car | 0.50 | 0.8085 | 0.8001 | 0.8042 | 12,832 | 3,040 | 3,207 |
| truck | 0.40 | 0.4397 | 0.3840 | 0.4100 | 288 | 367 | 462 |
| bus | 0.40 | 0.6218 | 0.5896 | 0.6053 | 148 | 90 | 103 |
| motorcycle | 0.35 | 0.6276 | 0.5395 | 0.5802 | 3,487 | 2,069 | 2,976 |

## Per-class results

### Confidence 0.05

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.2435 | 0.7827 | 0.3714 | 10,933 | 33,967 | 3,036 |
| bicycle | 0.1470 | 0.4569 | 0.2224 | 588 | 3,413 | 699 |
| car | 0.4065 | 0.9322 | 0.5661 | 14,951 | 21,828 | 1,088 |
| truck | 0.1622 | 0.5613 | 0.2517 | 421 | 2,174 | 329 |
| bus | 0.3511 | 0.6574 | 0.4577 | 165 | 305 | 86 |
| motorcycle | 0.2288 | 0.7756 | 0.3533 | 5,013 | 16,900 | 1,450 |

### Confidence 0.10

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.3135 | 0.7576 | 0.4434 | 10,583 | 23,180 | 3,386 |
| bicycle | 0.1942 | 0.4242 | 0.2664 | 546 | 2,266 | 741 |
| car | 0.4886 | 0.9221 | 0.6388 | 14,790 | 15,478 | 1,249 |
| truck | 0.2217 | 0.5333 | 0.3132 | 400 | 1,404 | 350 |
| bus | 0.4080 | 0.6534 | 0.5023 | 164 | 238 | 87 |
| motorcycle | 0.3030 | 0.7439 | 0.4306 | 4,808 | 11,062 | 1,655 |

### Confidence 0.15

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.3869 | 0.7275 | 0.5052 | 10,163 | 16,105 | 3,806 |
| bicycle | 0.2390 | 0.3901 | 0.2964 | 502 | 1,598 | 785 |
| car | 0.5563 | 0.9106 | 0.6906 | 14,605 | 11,651 | 1,434 |
| truck | 0.2709 | 0.5053 | 0.3527 | 379 | 1,020 | 371 |
| bus | 0.4568 | 0.6534 | 0.5377 | 164 | 195 | 87 |
| motorcycle | 0.3750 | 0.7039 | 0.4893 | 4,549 | 7,581 | 1,914 |

### Confidence 0.20

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.4665 | 0.6924 | 0.5574 | 9,672 | 11,061 | 4,297 |
| bicycle | 0.2846 | 0.3543 | 0.3157 | 456 | 1,146 | 831 |
| car | 0.6123 | 0.8996 | 0.7287 | 14,429 | 9,136 | 1,610 |
| truck | 0.3081 | 0.4827 | 0.3761 | 362 | 813 | 388 |
| bus | 0.4865 | 0.6454 | 0.5548 | 162 | 171 | 89 |
| motorcycle | 0.4459 | 0.6646 | 0.5337 | 4,295 | 5,337 | 2,168 |

### Confidence 0.25

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.5427 | 0.6566 | 0.5943 | 9,172 | 7,728 | 4,797 |
| bicycle | 0.3291 | 0.3201 | 0.3245 | 412 | 840 | 875 |
| car | 0.6567 | 0.8889 | 0.7553 | 14,257 | 7,454 | 1,782 |
| truck | 0.3520 | 0.4600 | 0.3988 | 345 | 635 | 405 |
| bus | 0.5096 | 0.6335 | 0.5648 | 159 | 153 | 92 |
| motorcycle | 0.5082 | 0.6240 | 0.5602 | 4,033 | 3,903 | 2,430 |

### Confidence 0.30

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6154 | 0.6171 | 0.6162 | 8,620 | 5,388 | 5,349 |
| bicycle | 0.3778 | 0.2859 | 0.3255 | 368 | 606 | 919 |
| car | 0.6976 | 0.8762 | 0.7768 | 14,054 | 6,091 | 1,985 |
| truck | 0.3803 | 0.4320 | 0.4045 | 324 | 528 | 426 |
| bus | 0.5306 | 0.6215 | 0.5725 | 156 | 138 | 95 |
| motorcycle | 0.5687 | 0.5821 | 0.5753 | 3,762 | 2,853 | 2,701 |

### Confidence 0.35

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6829 | 0.5758 | 0.6248 | 8,043 | 3,734 | 5,926 |
| bicycle | 0.4284 | 0.2510 | 0.3165 | 323 | 431 | 964 |
| car | 0.7323 | 0.8615 | 0.7917 | 13,818 | 5,051 | 2,221 |
| truck | 0.4064 | 0.4080 | 0.4072 | 306 | 447 | 444 |
| bus | 0.5698 | 0.6016 | 0.5853 | 151 | 114 | 100 |
| motorcycle | 0.6276 | 0.5395 | 0.5802 | 3,487 | 2,069 | 2,976 |

### Confidence 0.40

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.7452 | 0.5315 | 0.6205 | 7,425 | 2,539 | 6,544 |
| bicycle | 0.4870 | 0.2176 | 0.3008 | 280 | 295 | 1,007 |
| car | 0.7616 | 0.8441 | 0.8008 | 13,539 | 4,237 | 2,500 |
| truck | 0.4397 | 0.3840 | 0.4100 | 288 | 367 | 462 |
| bus | 0.6218 | 0.5896 | 0.6053 | 148 | 90 | 103 |
| motorcycle | 0.6726 | 0.4953 | 0.5705 | 3,201 | 1,558 | 3,262 |

### Confidence 0.50

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.8414 | 0.4161 | 0.5569 | 5,813 | 1,096 | 8,156 |
| bicycle | 0.5755 | 0.1422 | 0.2280 | 183 | 135 | 1,104 |
| car | 0.8085 | 0.8001 | 0.8042 | 12,832 | 3,040 | 3,207 |
| truck | 0.4980 | 0.3333 | 0.3994 | 250 | 252 | 500 |
| bus | 0.7097 | 0.5259 | 0.6041 | 132 | 54 | 119 |
| motorcycle | 0.7494 | 0.4062 | 0.5268 | 2,625 | 878 | 3,838 |

## Limitations

- The prediction artifact is already post-NMS and post-max_det, so thresholding cannot recover suppressed or truncated boxes.
- The selected threshold is tuned and measured on the same validation split; confirm it on untouched labeled data before deployment claims.
- F1 depends on the configured class-aware IoU criterion and is not COCO mAP or a replacement for the fixed-protocol metrics.
- The harness GT has no COCO crowd/ignore annotations; omitted VisDrone ignore regions can inflate apparent false positives.
- Per-class threshold selection has greater overfitting risk than one global threshold and requires held-out confirmation.
- Prediction scores are stored at finite precision in the evaluation artifact, so equality follows the serialized values.
