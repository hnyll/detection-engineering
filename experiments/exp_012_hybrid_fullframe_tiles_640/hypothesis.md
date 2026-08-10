# exp_012_hybrid_fullframe_tiles_640 — Hypothesis

## Question

Can one full-frame 960 view restore the context and large-object behavior lost
by Exp11's pure tiles while retaining its tiny-object scale benefit?

## Prediction (written before evaluation)

Exp11 strongly improved small-object AP and native tiny recall, but AP-large
fell from Exp10's 0.58783 to 0.41141. Exp12 adds the exact full-frame view to
the unchanged tile candidate pool before the unchanged global merge. The
full-frame candidates should restore context-complete large boxes, while tile
candidates preserve magnified tiny features.

All three co-primary criteria must pass:

1. AP-large is at least **0.55783**, the previously preregistered balanced
   deployment floor and a recovery of at least 0.14642 from Exp11.
2. AP-small is at least **0.22582**, retaining the Exp11 value within 0.010.
3. Native-shortest-side `<8 px` recall is at least **0.60973**, retaining the
   Exp11 value within 0.030.

AP-large from 0.49962 through 0.55782 is prespecified as partial context
recovery but still fails the balanced hybrid primary. Below 0.49962, the
full-frame context-recovery mechanism is not meaningfully supported. AP-large
contains only 1,068 GT boxes and is car-dominated, so it is interpreted with
the broader recall, mAP, and per-class checks below.

At least two supporting criteria must pass:

- AP-medium is at least **0.41011**, the prior Exp11 safety floor;
- native `>=32 px` recall is at least **0.93865**, halfway from Exp11 back to
  Exp10;
- mAP50-95 is at least **0.31628** (+0.005 versus Exp11);
- TIDE Loc is at most **4.357** (-0.250 dAP versus Exp11).

All accuracy guardrails must pass:

- mAP50-95 is at least 0.30628 and AR-max at least 0.44216;
- AP-medium is at least 0.39098 even if it misses the stronger supporting target;
- native `8-15 px` recall is at least 0.80177;
- no class AP50-95 declines by more than 0.010 from Exp11: person >=0.26616,
  bicycle >=0.10428, car >=0.57555, truck >=0.22154, bus >=0.37977, and
  motorcycle >=0.26037;
- TIDE Loc is at most 5.107, Miss at most 7.254, Bkg at most 3.726, Dupe at
  most 1.604, special FalsePos at most 19.320, and special FalseNeg at most
  19.277;
- no invalid or out-of-image final box is emitted;
- high-confidence top-300 pressure remains bounded: at most 109/548 images
  have rank-300 confidence at least 0.10 and at most 10/548 have rank-300
  confidence at least 0.25. Raw low-floor saturation is always reported and
  is not redefined as fixed.

Classification and Both dAP, mAP50, and mAP75 are reported diagnostics rather
than additional correlated gates. TIDE modes are nonlinear and nonadditive.

Deployment is evaluated separately. Mean end-to-end latency must be at most
105.5 ms/image (at least about 9.5 FPS) to remain a usable accuracy-oriented
candidate. A proportional efficiency expectation is at most 88.10 ms for the
additional full-frame forward, but missing that expectation alone does not
change the accuracy verdict.

Success requires all three co-primary criteria, at least two supporting
criteria, and every accuracy guardrail. If AP-large passes while retention or
pressure guardrails fail, record “context mechanism supported, fusion
negative.” If accuracy passes but latency fails, record “accuracy supported,
deployment negative.” These are engineering thresholds for one deterministic
checkpoint, not confidence intervals.

## Frozen control identity

Direct control: `exp_011_tiled_inference_640`, rerun under the finalized schema-2
pipeline before this hypothesis was evaluated:

- checkpoint SHA `d078cf0e25577202`;
- GT fingerprint `e1142868d4dc2e9d`;
- expected effective protocol hash `feddb3567d439c3a` under the frozen fusion-capable source;
- prediction SHA `2e324ed84dc1325f`;
- mAP/AP-small/AP-large `0.31128 / 0.23582 / 0.41141`;
- TIDE Cls/Loc/Miss `12.424 / 4.607 / 6.754`.

Exp10 is the full-frame deployment reference, not the direct causal control:
protocol `821306fe3a07ceac`, prediction SHA `310c355451e114e3`, using the same
checkpoint and GT.

The preregistered Exp12 hybrid protocol is expected to hash to
`051e7ff3d9dcb3e0`; evaluation must refuse or be rerun if the frozen source or
configuration changes.

## Causal variable changed

`evaluation.tiling.include_full_image`: **false -> true**.

The implementation reruns the identical Exp11 tile geometry and per-view
settings, appends one full-frame 960 candidate set before global fusion, then
applies the same class-aware NMS at IoU 0.70 and final top-300 cap. This is
deployment-equivalent early fusion, not late fusion of the rounded/capped
Exp10 and Exp11 JSON artifacts.

## Fixed inputs and excluded changes

Checkpoint, dataset, GT, taxonomy, validation split, tile size, overlap,
`imgsz`, confidence floor, per-view NMS, per-view cap, global merge, final cap,
tile batching, and evaluation implementation remain fixed. Exp12 performs no
training, threshold tuning, TTA, ignore-region correction, NMS sweep,
`max_det` increase, boundary filtering, or score weighting.

The extra forward and added candidate competition are inherent to the causal
change. Fusion may restore context while worsening duplicate, false-positive,
or cap pressure; those outcomes are explicit guardrails rather than silently
tuned away.
