---
exp: "exp_007_inference_960"
hypothesis_confirmed: false
outcome: positive
next_experiment: "exp_008_train_960"
---

# exp_007_inference_960 — Decision

## Problem

Exp6's tiny objects may lose meaningful class and boundary information when
full images are resized to 640. Exp7 tested whether the unchanged Exp6 model
performs better when inference alone retains 960 pixels of input resolution.

## Result

The resolution mechanism is strongly supported. With the exact same checkpoint
and ground truth, mAP50-95 increased from 0.21767 to 0.26661, AP-small from
0.11139 to 0.17284, mAP75 from 0.21391 to 0.27169, and AR-max from 0.29215 to
0.36598. Every class improved. TIDE classification, localization, misses, and
special FalseNeg impacts also decreased.

The gain came with a precision-side tradeoff. TIDE background impact rose from
2.219 to 3.070 and special FalsePos from 12.493 to 15.170. AP-large declined
from 0.51011 to 0.46163, and images reaching `max_det=300` increased from 458 to
476 of 548.

## Preregistered outcome

The primary AP-small criterion passed, as did two of the three required
supporting criteria (AR-max and TIDE Loc). The mAP, classification, and latency
guardrails passed. Special FalsePos exceeded its 14.493 guardrail by 0.677 dAP,
so the hypothesis is marked **false under the complete preregistered rule**.

The overall outcome is nevertheless **positive**: the same model gained 4.894
absolute mAP50-95 points and 6.145 AP-small points, with a large reduction in
false-negative impact. The useful conclusion is “higher-resolution inference
materially improves this checkpoint, with increased false-positive impact,”
not “all criteria passed.”

## Causal interpretation

Exp7 changed only prediction resolution from 640 to 960. It did not retrain the
network, change weights, tune confidence, tune NMS, tile images, or use test-time
augmentation. The paired evidence therefore supports effective resolution as
a real bottleneck for the Exp6 checkpoint.

The result belongs to a different inference operating point, whose effective
protocol hash is `821306fe3a07ceac`; it must not silently replace the fixed-640
result. Ordinary `make compare` correctly refuses Exp6 versus Exp7 because the
protocol hashes differ. The deliberate paired comparison is valid because the
checkpoint SHA (`eaed04c240d52e73`) and GT fingerprint (`e1142868d4dc2e9d`) are
identical and the changed resolution is explicitly recorded.

Both artifacts record git commit `e10f4e5`, but the dirty worktree means that
commit does not pin uncommitted evaluator changes. This is a provenance caveat,
not evidence of a measured mismatch; rerun Exp6 under the current harness if an
archival-grade code-identity proof becomes necessary.

## Tradeoffs

- Theoretical compute rose from 6.32 to 14.22 GFLOPs.
- Recorded mean latency rose from 10.06 to 10.77 ms/image, but this unexpectedly
  small difference should be remeasured before deployment decisions.
- More predictions reached the fixed `max_det` cap, which may constrain recall
  and complicate interpretation at the confidence floor of 0.001.
- Background and aggregate false-positive impacts increased.
- AP-large regressed even though small, medium, aggregate, and every class AP
  improved.

## Solution families considered

| Family | Candidate | Decision |
|---|---|---|
| Training resolution | Train the same YOLO11n setup at 960 and evaluate at 960 | Select for Exp8 |
| Threshold calibration | Select an operating confidence on validation data | Useful after Exp8; keep separate from training ablation |
| Detection cap | Raise `max_det` above 300 | Defer; changing it would create a second inference-protocol experiment |
| Tiling | Overlapping source-image tiles | Defer until full-image 960 training is measured |
| NMS tuning | Tune NMS IoU | Defer; duplicate dAP remains small |
| Label filtering | Remove tiny labels | Reject unless the application also ignores them during evaluation |

## Exp8 validation plan

Exp8 will start from the same `yolo11n.pt` pretrained model and use the same
six-class training data, repeat-factor manifest, seed 17, default augmentation,
and 100-epoch budget as Exp6. Training `imgsz` changes from 640 to 960, while
evaluation remains at 960 so Exp7 is the accuracy control. Batch size decreases
from 16 to 4 only because of the 8 GB VRAM limit; this compensation is recorded
as a limitation rather than hidden.

Threshold calibration, tiling, NMS changes, and `max_det` changes are excluded.
This lets Exp8 answer whether training at the intended inference resolution
adds value beyond simply evaluating the 640-trained model at 960.

## Recommendation

Use Exp7's 960 inference setting as the current high-accuracy operating point,
while retaining Exp6 at 640 when compute or false-positive tolerance matters.
Proceed to Exp8 training at 960. Do not claim a new trained model from Exp7 and
do not tune thresholds until the Exp8 comparison is complete.
