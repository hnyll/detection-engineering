# exp_007_inference_960 — Hypothesis

## Question

Does evaluating the unchanged Exp6 detector at `imgsz=960` retain enough
additional input detail to reduce tiny-object misses and localization errors?

## Prediction (written before evaluation)

AP-small will increase from 0.11139 to at least 0.12139. The result is accepted
only if at least two of the following also occur:

- AR-max increases from 0.29215 to at least 0.30215.
- TIDE Miss dAP decreases from 9.288 to at most 8.288.
- TIDE Loc dAP decreases from 6.252 to at most 5.752.

Guardrails:

- mAP50-95 remains at least 0.21267.
- TIDE Cls dAP does not exceed 15.879.
- special FalsePos dAP does not exceed 14.493.
- mean batch-1 latency does not exceed 25.15 ms under the recorded machine
  conditions.

Person, bicycle, and motorcycle AP are diagnostic secondary metrics. They are
not additional pass/fail criteria because their prevalence and difficulty are
different.

## Background evidence

Exp6's remaining main TIDE errors are classification 14.879 dAP, misses 9.288,
and localization 6.252. AP-small is 0.11139 versus 0.32413 medium and 0.51011
large. At the 640 input, median validation boxes are only 5.6 x 12.3 pixels for
person, 9.4 x 10.4 for bicycle, and 10.4 x 11.3 for motorcycle.

The confusion-matrix bottom background row contains unmatched ground truths,
not a background class emitted by YOLO. Its size and the low effective box
dimensions support an information/recall hypothesis. A 960 input has 1.5x the
linear dimension and 2.25x the square pixels of 640, although larger source
images are still downscaled and no new source detail is created.

Exp6 predictions reach `max_det=300` on 458 of 548 validation images at the
0.001 confidence floor. Exp7 keeps this cap fixed for causal control; saturation
will be noted if recall gains appear limited.

## Baseline

`exp_006_coco_aligned_balanced`, seed 17, evaluated at `imgsz=640`:

- checkpoint SHA: `eaed04c240d52e73`
- six-class GT fingerprint: `e1142868d4dc2e9d`
- mAP50-95: 0.21767
- AP-small: 0.11139
- AR-max: 0.29215
- mean latency: 10.06 ms/image

## Causal variable changed

Only inference `imgsz` changes from 640 to 960. The checkpoint, six-class
ground truth, confidence floor 0.001, NMS IoU 0.7, max detections 300, device,
batch 1, `rect=False`, prediction code, and COCO/TIDE evaluation remain fixed.
There is no training, tiling, test-time augmentation, or threshold tuning.

## Compensating implementation changes

The harness loads an experiment-local protocol override and inherited checkpoint
instead of modifying the global 640 protocol or copying weights. The effective
protocol receives a distinct hash, so ordinary fixed-protocol comparison must
refuse Exp6 versus Exp7. The paired conclusion will explicitly verify identical
checkpoint and ground-truth hashes.

## Interpretation boundary

If Exp7 passes, the claim is that the same model benefits from higher-resolution
inference. It is not evidence that training improved, and its result does not
replace the fixed-640 metric silently. Tiling and 960 training remain separate
future interventions.
