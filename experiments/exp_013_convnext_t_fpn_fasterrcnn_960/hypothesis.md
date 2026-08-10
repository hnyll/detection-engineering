# exp_013_convnext_t_fpn_fasterrcnn_960 — Preregistered Hypothesis

## Question

Does a pretrained ConvNeXt-Tiny + P2-P6 FPN + Faster R-CNN two-stage model
produce materially better object-level classification than the existing
full-frame YOLO11n system at 960?

This is a model-system screen, not a single-variable architecture ablation.
The backbone, detection paradigm, preprocessing, augmentation, optimizer,
physical batch, and epoch schedule necessarily change together. Exp10 seed 17
is the practical full-frame reference; Exp8 is the closer no-class-weighting
training reference. Ordinary fixed-protocol comparison must refuse the pair
because the preprocessing/postprocessing identities differ.

## Mechanism

The FPN supplies semantically enriched P2-P6 features for scale variation. The
RPN first separates objectness from category prediction, after which RoIAlign
gives the classifier an object-aligned feature. If dense one-stage candidate
competition or insufficient region-level semantic capacity is limiting Exp10,
TIDE Cls and clean truck/car and bicycle/motorcycle confusions should improve.

This design cannot reconstruct visual evidence absent from the source pixels.
Proposal recall may also become a new bottleneck for tiny or occluded objects.

## Frozen recipe

- ImageNet-pretrained ConvNeXt-Tiny; every trainable backbone/FPN/head parameter
  is fully fine-tuned.
- Native-aspect max-side-960 input; physical batch 1; accumulation 4; AMP.
- AdamW, learning rate 1e-4, weight decay 0.05, cosine decay, 24 epochs.
- Exact Exp6 six-class repeat-factor manifest and seed 17.
- P2-P6 anchor sizes 8/16/32/64/128; ratios 0.5/1/2.
- Confidence floor 0.001, class-aware NMS 0.70, at most 300 detections/image.
- Horizontal flip 0.5 only. No mosaic, generic augmentation additions, class
  weighting, tiling, threshold tuning, or context-refinement branch.

## Engineering screen criteria

Reference values are Exp10 seed 17: mAP50-95 0.29311, AP-small 0.17903,
TIDE Cls 14.225, Miss 8.300, Loc 4.850.

Both primary criteria must pass:

1. TIDE Cls <= 13.225 (at least 1.0 dAP lower).
2. mAP50-95 >= 0.30311 (at least +0.010).

At least two supporting criteria must pass:

- AP-small >= 0.18903.
- AR-max >= 0.39521.
- truck AP50-95 >= 0.24854.
- bicycle AP50-95 >= 0.08835.

All quality guardrails must pass:

- mAP50-95 >= 0.28811.
- no class AP50-95 declines more than 0.015 versus Exp10.
- TIDE Miss <= 8.800 and Loc <= 5.350.
- TIDE Bkg <= 3.813 and special FalsePos <= 15.811.
- AP-small >= 0.16903.

Deployment is judged separately: report batch-1 mean/p50/p95 latency and peak
VRAM. At least 10 FPS is the exploratory usability floor; failure does not
erase an accuracy mechanism result, but rejects it as a real-time candidate.

One seed is an engineering screen, not a repeatability or significance claim.
Run seeds 42 and 1337 only if the seed-17 screen passes both primaries without
violating guardrails.
