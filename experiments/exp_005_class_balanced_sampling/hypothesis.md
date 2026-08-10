# exp_005_class_balanced_sampling — Hypothesis

## Question

Does repeat-factor image sampling improve rare-class classification on the
official ten-class VisDrone task without materially reducing common-class or
overall AP?

## Prediction (written before training)

Relative to Exp1 seed 17, the mean AP50-95 of bicycle, truck, tricycle,
awning-tricycle, and bus will improve by at least 0.015; TIDE Cls dAP will
decrease by at least 1.5; and overall mAP50-95 will improve by at least 0.005.
Car AP50-95 must not decline by more than 0.010.

## Background reading / prior evidence

Classification is the largest TIDE error category in Exp1, Exp2, and Exp4.
The training labels are imbalanced: car has 144,866 boxes, while bicycle has
10,480, truck 12,875, tricycle 4,812, awning-tricycle 3,246, and bus 5,926.
Exp4's finer P2 representation improved small-object recall but increased
classification and false-positive impact, so this experiment targets the
measured class-frequency mechanism instead of feature resolution.

## Baseline

`exp_001_baseline_640`, seed 17. Exp2's resolution intervention did not meet its
preregistered threshold, so the 640px stock baseline is the cleaner parent for
an isolated sampling test.

## Causal variable changed

Only training-image sampling changes. A deterministic repeat-factor manifest
repeats images containing classes present in fewer than 80% of training images,
using `sqrt(0.80 / class_image_frequency)` capped at 3 repeats. The 0.80
threshold was selected before training from a manifest audit: 0.20 expanded the
epoch by only 1% and did not create a meaningful exposure change, while 0.80
expands it by approximately 56% and increases every non-dominant class. Model,
pretrained initialization, taxonomy, augmentations, image size, epoch budget,
validation data, evaluation protocol, and seed remain fixed.

## Compensating implementation changes

An epoch contains more manifest entries and therefore more optimizer steps,
training time, and repeated common-class objects from multi-class images. No
image or label files are copied, and validation remains the untouched official
VisDrone split.

## Why ATSS is deferred

ATSS changes positive-location assignment, not class balance. YOLO11 already
uses a task-aligned assigner, Exp4 localization dAP was relatively small, and
duplicate/NMS impact was low. Replacing the assigner would require a separate
custom training implementation and would confound this data-sampling test.
