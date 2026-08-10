# exp_008_train_960 — Hypothesis

## Question

Does training YOLO11n at the intended 960 inference resolution improve the
accuracy/false-positive tradeoff over evaluating the 640-trained Exp6
checkpoint at 960?

## Prediction (written before training)

Resolution-matched training will improve both overall and small-object AP over
Exp7 because the detector will learn its class confidence and box regression at
the same object scale used during evaluation.

Primary criteria, both required:

- mAP50-95 increases from 0.26661 to at least **0.27661** (+0.010).
- AP-small increases from 0.17284 to at least **0.18284** (+0.010).

At least two supporting criteria must also pass:

- AR-max increases from 0.36598 to at least **0.37598** (+0.010).
- TIDE Miss decreases from 8.655 to at most **8.155** (-0.500 dAP).
- TIDE Loc decreases from 5.086 to at most **4.586** (-0.500 dAP).
- Special FalsePos decreases from 15.170 to at most **14.170** (-1.000 dAP).

All guardrails must pass:

- AP-large remains at least **0.44163** (no further loss greater than 0.020).
- TIDE Cls remains at most **14.895** (no worsening greater than 1.000 dAP).
- Special FalsePos remains at most **16.170** (no worsening greater than 1.000
  dAP if its supporting improvement target is not reached).
- No individual class AP50-95 decreases by more than **0.010** from Exp7.
- Recorded mean batch-1 inference latency remains at most **13.50 ms/image**.

These are practical seed-17 screening thresholds, not statistical confidence
bounds. If the screen passes, additional training seeds are needed before a
repeatability claim.

## Background evidence

Exp7 reused Exp6's exact 640-trained checkpoint and changed only inference
resolution from 640 to 960. That increased mAP50-95 by 0.04894, AP-small by
0.06145, AR-max by 0.07383, and improved every class. TIDE localization and
special FalseNeg impacts also decreased substantially. This establishes that
the higher-resolution operating point exposes useful information.

Exp7 also increased TIDE special FalsePos from 12.493 to 15.170 and Bkg from
2.219 to 3.070, while AP-large fell by 0.04848. A plausible explanation is that
a model trained and checkpoint-selected at 640 is not optimally calibrated for
the scale distribution it sees at 960. Exp8 tests that explanation; it does not
assume it is already true.

## Baseline

The direct control is `exp_007_inference_960`, seed 17:

- training/evaluation sizes: 640/960
- mAP50-95: 0.26661
- AP-small: 0.17284
- AP-large: 0.46163
- AR-max: 0.36598
- TIDE Cls/Loc/Miss: 13.895 / 5.086 / 8.655 dAP
- TIDE special FalsePos/FalseNeg: 15.170 / 24.643 dAP
- mean latency: 10.77 ms/image
- evaluation protocol hash: `821306fe3a07ceac`
- six-class GT fingerprint: `e1142868d4dc2e9d`

Exp6 is useful historical context but is not the direct numerical control
because it was evaluated at 640.

## Causal variable changed

Only training `imgsz` changes from 640 to 960. Exp8 starts fresh from the same
stock `yolo11n.pt` rather than continuing the Exp6 checkpoint, then evaluates
its own `best.pt` at 960. The six-class taxonomy, exact 8,716-entry repeat-factor
manifest, seed 17, 100-epoch budget, default augmentation policy, architecture,
and evaluation settings remain fixed.

The pinned inputs are:

- stock `yolo11n.pt` SHA-256:
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- repeat-factor manifest SHA-256:
  `2b3a16c59f595c3bb51582b66849d9bc2dbebbce2eb0b5700edd292551ea74d3`

## Compensating implementation changes

Physical batch size decreases from 16 to 4 for the 8 GB RTX 4060. Exp2 already
completed at 1024 with batch 4, making this the evidence-based safe setting.
Ultralytics' `nbs=64` gradient accumulation preserves the nominal accumulated
batch, but the smaller microbatch changes BatchNorm statistics and the exact
optimization trajectory. Therefore the experiment is not a mathematically
perfect one-variable ablation; that hardware compensation is part of the
interpretation boundary.

Exp8's `best.pt` is selected using 960 internal validation, whereas Exp7
inherits a checkpoint selected at 640. That checkpoint-selection behavior is
part of the resolution-matched training pipeline being tested.

## Excluded changes

Exp8 does not tune confidence, NMS IoU, `max_det`, augmentation, sampling,
taxonomy, architecture, loss weights, tiling, or test-time augmentation. The
fixed `max_det=300` remains highly saturated, but changing it here would break
the direct Exp7 comparison and is reserved for a separate experiment.
