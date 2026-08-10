# exp_011_tiled_inference_640 — Failure Analysis

Exp11 is a paired inference-pipeline diagnostic against Exp10 seed 17. It
reuses the exact trained checkpoint and six-class validation GT, replacing one
full-frame pass with deterministic 640x640 source tiles at 20% overlap. Every
tile is inferred at `imgsz=960`; mapped detections are merged with class-aware
NMS at IoU 0.70 and capped at 300 per original image. No training or manual
FiftyOne review was performed for Exp11.

TIDE values below are oracle AP impacts (dAP), not error counts. Its main
categories are independently corrected and are not additive. Special
FalsePos/FalseNeg are an overlapping alternative decomposition.

## Provenance and comparison boundary

| Identity | Exp10 control | Exp11 tiled | Check |
|---|---|---|---|
| checkpoint SHA | `d078cf0e25577202` | `d078cf0e25577202` | identical |
| dataset / split | `visdrone_exp006_coco_aligned` / val | same | identical |
| GT fingerprint | `e1142868d4dc2e9d` | `e1142868d4dc2e9d` | identical |
| imgsz / conf / NMS IoU / final max_det | 960 / 0.001 / 0.70 / 300 | same | identical |
| inference pipeline | one full-frame pass | pure 640x640 tiles, 20% overlap, global merge | intended treatment |
| effective protocol hash | `821306fe3a07ceac` | `feddb3567d439c3a` | intentionally different |
| Exp11 prediction SHA | — | `2e324ed84dc1325f` | matches TIDE |

Ordinary fixed-protocol comparison must reject Exp10 versus Exp11 because the
inference pipeline is the intervention and therefore has a distinct protocol
identity. A deliberate paired interpretation is valid because checkpoint, GT,
scalar inference settings, and evaluation code are held fixed. It describes
this checkpoint on this validation set, not repeatability across training seeds
or an unbiased benchmark result.

The tiled engine also passed view-preprocessing parity on one image from each
of the three native shapes: file-path and ndarray inference produced identical
counts, classes, boxes, and scores. The explicit global merge then removed two
of 294 boxes on one uncapped 960x540 image; that extra merge is part of the
registered tiled treatment rather than hidden preprocessing drift.

## Preregistered outcome

| Criterion | Exp10 | Required Exp11 | Exp11 | Delta | Result |
|---|---:|---:|---:|---:|---|
| AP-small, co-primary | 0.17903 | >=0.18903 | 0.23582 | +0.05679 | pass |
| native `<8 px` recall, co-primary | 0.40717 | >=0.45720 | 0.63973 | +0.23256 | pass |
| native `8-15 px` recall, supporting | 0.70744 | >=0.73740 | 0.82177 | +0.11433 | pass |
| AR-max, supporting | 0.38521 | >=0.39521 | 0.45216 | +0.06695 | pass |
| TIDE Miss dAP, supporting | 8.300 | <=7.800 | 6.754 | -1.546 | pass |
| TIDE Loc dAP, supporting | 4.850 | <=4.600 | 4.607 | -0.243 | fail by 0.007 |
| mAP50-95 guardrail | 0.29311 | >=0.28811 | 0.31128 | +0.01817 | pass |
| no class AP decline >0.010 | — | worst delta >=-0.010 | truck -0.00700 | — | pass |
| AP-medium guardrail | 0.42011 | >=0.41011 | 0.40098 | -0.01913 | fail |
| AP-large guardrail | 0.58783 | >=0.55783 | 0.41141 | -0.17642 | fail |
| TIDE Bkg dAP guardrail | 2.813 | <=3.813 | 3.226 | +0.413 | pass |
| TIDE Dupe dAP guardrail | 0.604 | <=1.104 | 1.604 | +1.000 | fail |
| special FalsePos dAP guardrail | 13.811 | <=15.811 | 19.320 | +5.509 | fail |
| special FalseNeg dAP guardrail | 25.865 | <=26.865 | 18.277 | -7.588 | pass |
| final invalid boxes | 0 | 0 | 0 | 0 | pass |
| images at global max_det | 85.04% | <=95% | 97.81% | +12.77 pp | fail |
| mean end-to-end latency | 10.55 ms | <=105.5 ms | 66.40 ms | +55.85 ms | pass |

Both co-primary criteria and three of four supporting criteria passed. However,
the preregistration required every guardrail to pass; AP-medium, AP-large,
Dupe, special FalsePos, and low-floor cap saturation failed. The strict
composite hypothesis is therefore not confirmed. The size mechanism itself is
strongly supported, while this exact pure-tile pipeline is not an unconditional
deployment replacement.

## Accuracy and scale outcomes

| Metric | Exp10 full frame | Exp11 tiles | Delta |
|---|---:|---:|---:|
| mAP50-95 | 0.29311 | 0.31128 | +0.01817 |
| mAP50 | 0.48070 | 0.51307 | +0.03237 |
| mAP75 | 0.29885 | 0.31635 | +0.01750 |
| AP-small | 0.17903 | 0.23582 | +0.05679 |
| AP-medium | 0.42011 | 0.40098 | -0.01913 |
| AP-large | 0.58783 | 0.41141 | -0.17642 |
| AR-max | 0.38521 | 0.45216 | +0.06695 |

The 0.05679 AP-small gain and 0.23256 absolute gain in `<8 px` recall are large
for a frozen checkpoint. This directly supports effective feature scale as a
major limitation. Overall mAP also rises by 0.01817 despite the severe
AP-large regression. Pure crops magnify tiny objects, but they remove global
context and can split large objects across crop boundaries; those are plausible
causes of the medium/large tradeoff, not separately isolated conclusions.

### Native-source shortest-side recall

| Native shortest side | GT | Exp10 recall | Exp11 recall | Delta |
|---|---:|---:|---:|---:|
| `<8 px` | 5,074 | 0.40717 | 0.63973 | +0.23256 |
| `8-15 px` | 11,861 | 0.70744 | 0.82177 | +0.11433 |
| `16-31 px` | 13,414 | — | 0.87498 | — |
| `>=32 px` | 8,410 | 0.94400 | 0.93329 | -0.01071 |
| all | 38,759 | 0.76787 | 0.84055 | +0.07268 |

Recall here is a same-class, score-greedy IoU-0.50 diagnostic over the exported
candidate set. It is not COCO AP and does not replace TIDE.

## Per-class outcomes

| Class | Exp10 AP50-95 | Exp11 AP50-95 | Delta |
|---|---:|---:|---:|
| person | 0.22575 | 0.27616 | +0.05041 |
| bicycle | 0.07835 | 0.11428 | +0.03593 |
| car | 0.59110 | 0.58555 | -0.00555 |
| truck | 0.23854 | 0.23154 | -0.00700 |
| bus | 0.38872 | 0.38977 | +0.00105 |
| motorcycle | 0.23618 | 0.27037 | +0.03419 |

The largest gains occur in person, bicycle, and motorcycle, the classes most
affected by tiny targets. Car and truck decline slightly but remain within the
preregistered per-class guardrail.

## TIDE outcomes

| Error | Exp10 dAP | Exp11 dAP | Delta | Interpretation |
|---|---:|---:|---:|---|
| Cls | 14.225 | 12.424 | -1.801 | strong improvement; still largest main mode |
| Loc | 4.850 | 4.607 | -0.243 | directional improvement; target missed by 0.007 |
| Both | 0.647 | 0.899 | +0.252 | worse |
| Dupe | 0.604 | 1.604 | +1.000 | worse; overlap/merge pressure |
| Bkg | 2.813 | 3.226 | +0.413 | worse, but inside guardrail |
| Miss | 8.300 | 6.754 | -1.546 | strong improvement |
| special FalsePos | 13.811 | 19.320 | +5.509 | substantially worse |
| special FalseNeg | 25.865 | 18.277 | -7.588 | substantially better |

Tiling improves more than recall: Cls dAP falls by 1.801. This suggests that a
meaningful part of the apparent classification bottleneck was coupled to weak
feature scale or surrounding clutter rather than class frequency alone. It
does not prove that every wrong-class prediction was fixed, because TIDE is a
nonlinear oracle analysis and the candidate set changed substantially.

The opposite FalseNeg/FalsePos movement is a normal recall tradeoff here: tiles
recover more true objects while also producing more overlapping and background
candidates. Special FalsePos/FalseNeg must not be added to the main categories.

## Pipeline and deployment diagnostics

- 2,842 tiles were processed for 548 images: mean 5.186, range 2-8.
- Candidate counts were 740,183 before global merge, 488,344 after merge, and
  163,320 after the top-300 cap.
- 536/548 images (97.81%) reached the final 300-box cap at confidence 0.001;
  78 remained capped with rank-300 score at least 0.10 and 7 at least 0.25.
- The median rank-300 score was 0.0301, so much—but not all—of the saturation
  is in the low-confidence tail used for AP evaluation.
- 1,985/2,842 local tile passes (69.85%) also reached their local 300-box cap,
  but their median rank-300 score was only 0.00504; 34 were capped at 0.10 and
  none at 0.25. Local saturation is therefore mainly low-ranked AP-tail
  pressure, while global competition remains visible at higher confidence.
- Five invalid intermediate candidates were rejected; no invalid final boxes
  were emitted.
- Estimated detector compute rose from 14.22 to 73.73 GFLOPs per original
  image. End-to-end mean latency was 66.40 ms, with p50 74.77 ms and p95 81.68
  ms: approximately 15.1 FPS on the recorded RTX 4060.

Latency includes decode, slicing, all tile passes, coordinate mapping, and
global merge over 100 evenly spaced validation images. It passed the deliberately
loose 105.5-ms diagnostic guardrail. The recorded means suggest roughly a
6.5-times slowdown from Exp10, but that ratio is approximate because Exp10's
legacy timing used the first 100 sorted images rather than the same evenly
spaced sample.

## Threshold calibration diagnostic

The frozen tiled predictions were swept over confidence 0.05-0.50 at
same-class IoU 0.50. Within this candidate grid, confidence 0.40 maximized both
macro F1 (0.5513) and micro F1 (0.6842): micro precision 0.7325, recall 0.6419,
and 16.58 FP/image. At confidence 0.25, recall rises to 0.7322 but precision
falls to 0.5781 and FP/image rises to 37.80. At 0.10, recall is 0.8073 but
FP/image reaches 97.86.

This is operating-point calibration on the same validation set, not a model or
AP improvement. A deployment threshold must be chosen against an application
false-alarm budget and confirmed on untouched labeled data. Per-class optima
range from 0.30 to 0.50, but using them increases overfitting and maintenance
risk.

## Remaining failure modes

1. Classification opportunity remains the largest main TIDE mode at 12.424
   dAP, although tiling reduced it substantially.
2. Miss remains material at 6.754 dAP; magnification cannot create source
   evidence absent through extreme size or occlusion.
3. Localization remains material at 4.607 dAP, especially because small-box
   IoU is highly sensitive to a few pixels of displacement.

For this pipeline, large-object/context loss and candidate proliferation are
equally important engineering failures even though they are not additional
additive TIDE modes. The current GT also retains the known omission of original
VisDrone ignore regions, so absolute Bkg/FalsePos interpretation remains partly
contaminated; the omission is held fixed and does not explain the paired tiny
recall gain.

## Conclusion

Exp11 establishes that effective object scale was bottlenecking Exp10. Pure
tiling recovered many tiny targets, improved every aggregate AP except the
medium/large size bins, raised overall mAP, and reduced Cls, Miss, Loc, and
special FalseNeg opportunity. It also lost large-object performance, increased
duplicate and false-positive pressure, saturated the global detection budget,
and multiplied latency.

The correct conclusion is not “tiling failed” or “Exp11 replaces Exp10.” It is
“the tiny-object mechanism is strongly supported, while this pure-tile merge is
an accuracy/latency tradeoff that needs a context-preserving follow-up.”
