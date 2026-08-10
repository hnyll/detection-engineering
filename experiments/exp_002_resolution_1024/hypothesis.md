# exp_002_resolution_1024 — Hypothesis

## Question
<!-- What are we trying to learn? One sentence. -->
did the model improve after training with imgsz 1024 vs 640?

## Prediction (write BEFORE training)
<!-- Directional and quantified: "raising imgsz to 1024 will raise AP_small by ~3 points
     because tiny objects currently span <8 px on the P3 feature map". -->
i think that all metrics would improve in general but it wont be significant.

## Background reading / prior evidence
<!-- Papers, docs, previous experiments motivating this. -->
bigger input means more information/data it can use. might not be useful tho if orig img resolution is smaller than 1024 tho

## Baseline
<!-- Which experiment is the control? (meta.parent_exp in config.yaml) -->
exp_001 seed 17

## Causal variable changed
<!-- Exactly ONE. Everything else stays fixed. -->
imgsz from 640 to 1024

## Compensating implementation changes
<!-- Changes forced by the causal change (e.g., batch 16 -> 8 due to VRAM at 1024px).
     Documented separately per the guiding rules — they are NOT the thing under test. -->
batch size 8 -> 4 bc gpu mem limits