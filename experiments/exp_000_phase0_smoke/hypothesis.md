# exp_000_phase0_smoke — Hypothesis

## Question
Does every stage of the harness (train, resume, W&B, eval, TIDE, FiftyOne,
export, parity) run end-to-end on this machine?

## Prediction (write BEFORE training)
All stages complete without manual intervention; peak RSS stays under 7GB;
resume continues from epoch 2 after the planned interrupt at epoch 1.

## Background reading / prior evidence
docs/harness_spec.md Phase 0: COCO128 proves plumbing only — 3 epochs of
yolo11n cannot support any modeling conclusion.

## Baseline
None — this is the plumbing baseline.

## Causal variable changed
None (no hypothesis about the model is being tested).

## Compensating implementation changes
None.
