# exp_012_hybrid_fullframe_tiles_640 — Failure Analysis

Exp12 is a paired inference-only comparison against Exp11. It uses the same
Exp10 seed-17 checkpoint, validation GT, tiles, scalar inference settings, and
global merge, adding exactly one full-frame 960 view per source image. No
training or manual FiftyOne review was performed.

TIDE values are oracle AP impacts, not counts. Main categories are nonlinear
and nonadditive; special FalsePos/FalseNeg overlap the main decomposition.

## Provenance

| Identity | Exp10 full frame | Exp11 pure tiles | Exp12 hybrid | Check |
|---|---|---|---|---|
| checkpoint SHA | `d078cf0e25577202` | same | same | pass |
| GT fingerprint | `e1142868d4dc2e9d` | same | same | pass |
| protocol hash | `821306fe3a07ceac` | `feddb3567d439c3a` | `051e7ff3d9dcb3e0` | intended pipeline identities |
| prediction SHA | `310c355451e114e3` | `2e324ed84dc1325f` | `a532e51fa6247ab0` | recorded |
| tile-layout SHA | — | `79e86ef06d0c9009` | `79e86ef06d0c9009` | identical |
| pipeline | one full-frame view | 640 tiles only | same tiles + one full-frame view | intended treatment |

The frozen preregistration matches the evaluated hypothesis, configuration,
source implementation, checkpoint, GT, control predictions, and expected Exp12
protocol hash. View-preprocessing parity passed on all three native image
shapes. The Exp11 rerun under the frozen implementation reproduced its earlier
prediction SHA exactly.

The frozen evaluator source SHA is
`eff17e3fed47ac67ac18362fdb2d054e565c08618c80a0c7558701aa903efd17`;
the full Exp12 prediction SHA is
`a532e51fa6247ab0ad9c43dfb134f5b5f827fd71bab2e330882487a191707d92`.

Ordinary `make compare` must refuse these experiments because their prediction
pipelines have different protocol hashes. The paired interpretation is
deliberate: Exp11 is the direct causal control, while Exp10 is the full-frame
deployment reference.

## Preregistered outcome

### Co-primary and supporting criteria

| Type | Criterion | Required | Exp12 | Delta vs Exp11 | Result |
|---|---|---:|---:|---:|---|
| primary | AP-large | >=0.55783 | 0.54461 | +0.13320 | **fail by 0.01322** |
| primary | AP-small | >=0.22582 | 0.23629 | +0.00047 | pass |
| primary | native `<8 px` recall | >=0.60973 | 0.63106 | -0.00867 | pass |
| supporting | AP-medium | >=0.41011 | 0.42302 | +0.02204 | pass |
| supporting | native `>=32 px` recall | >=0.93865 | 0.94935 | +0.01606 | pass |
| supporting | mAP50-95 | >=0.31628 | 0.32360 | +0.01232 | pass |
| supporting | TIDE Loc | <=4.357 | 4.147 | -0.460 | pass |

Two of three co-primary criteria passed, and all four supporting criteria
passed. AP-large lies in the preregistered partial-recovery band but misses the
balanced-hybrid primary floor.

### Guardrails

| Guardrail | Required | Exp12 | Result |
|---|---:|---:|---|
| mAP50-95 | >=0.30628 | 0.32360 | pass |
| AR-max | >=0.44216 | 0.46513 | pass |
| AP-medium | >=0.39098 | 0.42302 | pass |
| native `8-15 px` recall | >=0.80177 | 0.81730 | pass |
| worst per-class AP decline | >=-0.010 | person -0.00028 | pass |
| TIDE Loc | <=5.107 | 4.147 | pass |
| TIDE Miss | <=7.254 | 6.799 | pass |
| TIDE Bkg | <=3.726 | 3.046 | pass |
| TIDE Dupe | <=1.604 | 1.976 | **fail** |
| special FalsePos | <=19.320 | 19.762 | **fail** |
| special FalseNeg | <=19.277 | 17.648 | pass |
| invalid final boxes | 0 | 0 | pass |
| rank-300 score >=0.10 | <=109 images | 118 | **fail** |
| rank-300 score >=0.25 | <=10 images | 14 | **fail** |
| mean latency | <=105.5 ms | 75.30 ms | pass |
| proportional latency expectation | <=88.10 ms | 75.30 ms | pass |

The strict success rule required all three primary criteria and every accuracy
guardrail. It therefore fails because AP-large, Dupe, FalsePos, and both
high-confidence cap-pressure limits failed.

## Accuracy and scale outcomes

| Metric | Exp10 | Exp11 | Exp12 | Exp12 - Exp11 |
|---|---:|---:|---:|---:|
| mAP50-95 | 0.29311 | 0.31128 | 0.32360 | +0.01232 |
| mAP50 | 0.48070 | 0.51307 | 0.52186 | +0.00879 |
| mAP75 | 0.29885 | 0.31635 | 0.33477 | +0.01842 |
| AP-small | 0.17903 | 0.23582 | 0.23629 | +0.00047 |
| AP-medium | 0.42011 | 0.40098 | 0.42302 | +0.02204 |
| AP-large | 0.58783 | 0.41141 | 0.54461 | +0.13320 |
| AR-max | 0.38521 | 0.45216 | 0.46513 | +0.01297 |

The added full-frame view recovers 75.5% of Exp11's 0.17642 AP-large deficit
to Exp10. AP-medium exceeds Exp10, and AP-small is retained rather than traded
away. Exp12 also has the highest aggregate mAP of the three pipelines.

Native `>=32 px` recall rises above Exp11 while `<8 px` and `8-15 px` recall
decline only 0.00867 and 0.00447 and remain inside their retention limits. The
strong large-object IoU-0.50 recall together with the remaining AP-large gap
shows that candidate existence is not the whole issue: ranking, precision, and
high-IoU box quality also matter.

### Native-source shortest-side recall

| Native shortest side | GT | Exp11 | Exp12 | Delta |
|---|---:|---:|---:|---:|
| `<8 px` | 5,074 | 0.63973 | 0.63106 | -0.00867 |
| `8-15 px` | 11,861 | 0.82177 | 0.81730 | -0.00447 |
| `16-31 px` | 13,414 | 0.87498 | 0.88229 | +0.00731 |
| `>=32 px` | 8,410 | 0.93329 | 0.94935 | +0.01606 |
| all | 38,759 | 0.84055 | 0.84406 | +0.00351 |

This is a same-class, score-greedy IoU-0.50 diagnostic over exported
candidates. It is not COCO AP and does not replace TIDE.

## Per-class AP50-95

| Class | Exp10 | Exp11 | Exp12 | Exp12 - Exp11 |
|---|---:|---:|---:|---:|
| person | 0.22575 | 0.27616 | 0.27588 | -0.00028 |
| bicycle | 0.07835 | 0.11428 | 0.12073 | +0.00645 |
| car | 0.59110 | 0.58555 | 0.59204 | +0.00649 |
| truck | 0.23854 | 0.23154 | 0.25609 | +0.02455 |
| bus | 0.38872 | 0.38977 | 0.42348 | +0.03371 |
| motorcycle | 0.23618 | 0.27037 | 0.27340 | +0.00303 |

Five classes improve over Exp11, and person is effectively unchanged. All six
classes exceed Exp10, although this remains a custom six-class task rather than
an official ten-class VisDrone benchmark.

## TIDE outcomes

| Error | Exp10 | Exp11 | Exp12 | Exp12 - Exp11 |
|---|---:|---:|---:|---:|
| Cls | 14.225 | 12.424 | 12.044 | -0.380 |
| Loc | 4.850 | 4.607 | 4.147 | -0.460 |
| Both | 0.647 | 0.899 | 0.833 | -0.066 |
| Dupe | 0.604 | 1.604 | 1.976 | +0.372 |
| Bkg | 2.813 | 3.226 | 3.046 | -0.180 |
| Miss | 8.300 | 6.754 | 6.799 | +0.045 |
| special FalsePos | 13.811 | 19.320 | 19.762 | +0.442 |
| special FalseNeg | 25.865 | 18.277 | 17.648 | -0.629 |

The full-frame view improves Cls, Loc, Both, Bkg, and FalseNeg relative to
Exp11. Miss is nearly unchanged. Dupe and special FalsePos worsen, confirming
that fusion candidate management—not feature visibility alone—is now the main
pipeline-specific limitation.

The three largest remaining main TIDE opportunities are Cls 12.044, Miss
6.799, and Loc 4.147 dAP. They should not be summed with special FalsePos or
FalseNeg.

## Fusion and deployment diagnostics

- Exp12 performs 6.186 forwards/image versus 5.186 for Exp11.
- Estimated detector compute rises from 73.73 to 87.94 GFLOPs/image.
- The full-frame pass contributes 153,153 valid candidates before global
  fusion. This differs from Exp10's serialized total by three zero-height,
  low-score boxes rejected by the frozen sanitizer, not by model inference.
- After fusion and the final cap, 115,235 tile-origin and 48,673 full-frame-
  origin detections survive; every image retains at least one full-frame box.
- Pre-merge candidates rise from 740,183 to 893,336, while final detections rise
  only from 163,320 to 163,908. Fusion changes ranking and source composition
  far more than the final count.
- Raw top-300 saturation rises from 536/548 images to 544/548.
- The median rank-300 score rises from 0.03010 to 0.04531.
- Images capped with rank-300 score at least 0.10 rise from 78 to 118; at 0.25,
  they rise from 7 to 14.
- No invalid final box is emitted.
- Mean latency is 75.30 ms, p50 83.12 ms, and p95 94.98 ms, or about 13.3 FPS.
  The additional view costs 8.90 ms over Exp11 and passes both latency gates.

## Threshold diagnostic

Confidence 0.40 maximizes both tested macro F1 (0.5567) and micro F1 (0.6768).
At that point, micro precision is 0.6854, recall 0.6685, and FP/image 21.71.

Relative to Exp11 at the same 0.40 threshold, Exp12 raises recall but lowers
precision and increases FP/image from 16.58 to 21.71. This agrees with the TIDE
and cap-pressure failures. The threshold was selected and measured on the same
validation split; it is not an AP improvement and must be confirmed on
untouched labeled data before deployment.

## Interpretation

The context-recovery mechanism is partially supported. Adding the full-frame
view recovers most of AP-large, fully repairs AP-medium, improves overall AP,
AR, localization, and five classes, while retaining the tiny-object gain.
However, it does not reach the preregistered AP-large floor and worsens
duplicate, false-positive, and high-confidence cap pressure.

The practical result is therefore positive as an accuracy experiment but
negative under the strict balanced-fusion acceptance rule. Exp12 is a stronger
accuracy/recall pipeline than Exp11, not yet an unconditional replacement for
the faster and lower-FP Exp10 pipeline.

These findings describe one deterministic checkpoint on one validation split.
They do not establish training-seed repeatability or statistical significance.
The omitted VisDrone ignore regions remain a fixed limitation of absolute
Bkg/FalsePos interpretation. The evaluator source and experiment artifacts are
currently uncommitted; their recorded implementation and artifact hashes bind
this result, but the tree should be committed before a publication handoff.

## Conclusion

Exp12 demonstrates that full-frame context and tiled detail are complementary.
The hybrid reaches 0.32360 mAP50-95, preserves AP-small, recovers most
large-object performance, and passes latency. Its remaining problem is the
candidate-fusion tradeoff: too many overlapping and false-positive candidates
compete for the fixed top-300 budget. The next experiment should isolate
candidate admission rather than change training, NMS, confidence, tile size,
and detection budget together.
