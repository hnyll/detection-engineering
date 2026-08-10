---
exp: "exp_012_hybrid_fullframe_tiles_640"
hypothesis_confirmed: false
outcome: positive
next_experiment: ""
---

# exp_012_hybrid_fullframe_tiles_640 — Decision

## Problem

Exp11 showed that tiled inference materially improves tiny-object detection,
but pure tiles lose full-frame context and large-object performance. Exp12
tested whether adding one full-frame view to the unchanged tile pool repairs
that tradeoff.

## Result

The hybrid improves mAP50-95 from 0.31128 to 0.32360, AP-medium from 0.40098
to 0.42302, AP-large from 0.41141 to 0.54461, and AR from 0.45216 to 0.46513.
AP-small is retained at 0.23629, and native `<8 px` recall remains high at
0.63106. Five classes improve over Exp11, and all six exceed Exp10.

The added view also increases candidate competition. TIDE Dupe rises from
1.604 to 1.976 dAP and special FalsePos from 19.320 to 19.762. High-confidence
top-300 pressure fails both registered limits. Mean latency is 75.30 ms, or
13.3 FPS, and passes the deployment gate.

## Preregistered decision

Two of three co-primary criteria pass. AP-large gains 0.13320, recovering
75.5% of the Exp11-to-Exp10 deficit, and lies in the preregistered partial band,
but it misses the 0.55783 primary floor by 0.01322. All four supporting
criteria pass. Dupe, FalsePos, and both high-confidence cap-pressure
guardrails fail.

Because success required every primary criterion and guardrail,
`hypothesis_confirmed` is `false`. The outcome is nevertheless `positive`:
the paired intervention produces the best observed aggregate accuracy,
recovers most context performance, retains tiny-object performance, and adds
only 8.90 ms over Exp11. Positive does not mean the fusion pipeline passed its
balanced-deployment acceptance rule.

## What the experiment establishes

1. Full-frame context and magnified tile detail are complementary.
2. The AP-large loss in Exp11 was primarily caused by the pure-tile pipeline,
   not a change in checkpoint quality.
3. Tiny-object gains can survive the addition of a context-complete view.
4. Candidate admission and fusion pressure are now more urgent than another
   resolution, loss-weight, or broad augmentation experiment.
5. Strong `>=32 px` recall with lower-than-target AP-large means candidate
   existence is not the whole problem; ranking, precision, and high-IoU box
   quality remain relevant.

AP-large contains only 1,068 validation GT boxes and 894 are cars. It is still
the correct primary because it measures the Exp11 regression, but it should
not be generalized as equal evidence for every class.

## Practical model choice

| Pipeline | Best use | Main limitation |
|---|---|---|
| Exp10 full frame | speed and lower false-positive pressure | weaker tiny recall and AP-small |
| Exp11 pure tiles | mechanism ablation; no longer preferred operationally | large-object/context loss |
| Exp12 hybrid | best accuracy and recall when approximately 13 FPS is acceptable | duplicate, FP, and top-300 pressure |

Exp12 can be retained as an accuracy-oriented experimental candidate. Exp10
remains the safer default where throughput or false alarms matter more. Exp11
should remain as causal evidence rather than the preferred deployment path.

## Operating threshold

Confidence 0.40 is the best tested global F1 point for Exp12, with micro
precision 0.6854, recall 0.6685, F1 0.6768, and 21.71 FP/image. This is a
validation-tuned operating point, not an AP improvement. It must be confirmed
on untouched labeled data and chosen against an application-specific false-
alarm budget.

## Stop condition and future work

Close the current experiment sequence at Exp12. The series has already tested
resolution, feature-pyramid changes, taxonomy and sampling, class weighting,
pure tiled inference, and context-preserving hybrid inference. Another run is
not necessary to support the current conclusions.

If the project is reopened, treat it as a new comparison study rather than an
extension required to validate Exp12. The most defensible options are:

- `YOLO11s` with the Exp8/Exp10 recipe as a capacity comparison. This tests a
  larger model in the same architecture family, not a new architecture.
- A transformer-based detector such as RT-DETR as a genuinely different
  architecture and context-modeling comparison. Match the taxonomy, split,
  evaluation protocol, input size, and deployment timing before comparing it.
- Tile-ownership filtering as a deployment-pipeline cleanup if duplicate and
  false-positive pressure becomes operationally important.

Do not run several architectures and report only the best validation result.
Any future comparison should preregister one model, its compute budget, and
the required gain relative to its latency and memory cost.

The known omitted VisDrone ignore regions should also be corrected in a
separate versioned evaluation protocol before making absolute FP/Bkg claims;
do not bundle that scoring change into a future model or pipeline comparison.

## Recommendation

Do not reject the hybrid mechanism: its accuracy gains are broad and
practically meaningful. Also do not call Exp12 a completed balanced deployment
pipeline because it missed the registered AP-large target and worsened
duplicate, false-positive, and cap pressure.

Keep Exp12 as the final accuracy-oriented six-class candidate and Exp10 as the
efficient full-frame reference. Record alternative architectures as future
work; no further experiment is required for the current project.
