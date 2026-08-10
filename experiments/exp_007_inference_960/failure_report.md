# exp_007_inference_960 — Failure Analysis

Exp7 is an inference-only experiment. It evaluates the exact Exp6 seed-17
checkpoint at `imgsz=960` instead of `imgsz=640`; it does not train or tune a
new model. The checkpoint SHA (`eaed04c240d52e73`), six-class ground-truth
fingerprint (`e1142868d4dc2e9d`), confidence floor (`0.001`), NMS IoU (`0.7`),
and `max_det` (`300`) are unchanged. The effective protocol hash changes from
`a9b8207b3cc7a070` to `821306fe3a07ceac` because inference resolution is the
intended intervention.

Both metric artifacts record git commit `e10f4e5`, but the repository contains
uncommitted work, so that commit alone is not a cryptographic record of the
evaluator source at each run. The checkpoint, GT, prediction, and effective
protocol identities are pinned. Re-running Exp6 evaluation with the current
harness would be the stricter archival check if that level of provenance is
later required.

No Exp7 FiftyOne review was performed, so this report does not claim manual
failure counts or condition prevalence. TIDE values are oracle AP impacts, not
counts. Its special FalsePos and FalseNeg categories overlap/reorganize the main
decomposition and must not be added to it.

## Paired Exp6/Exp7 results

| Metric | Exp6 at 640 | Exp7 at 960 | Delta | Interpretation |
|---|---:|---:|---:|---|
| mAP50-95 | 0.21767 | 0.26661 | +0.04894 | Large overall gain |
| mAP50 | 0.37672 | 0.44897 | +0.07225 | Better detection and coarse localization |
| mAP75 | 0.21391 | 0.27169 | +0.05778 | Better stricter localization |
| AP-small | 0.11139 | 0.17284 | +0.06145 | Primary target passed |
| AP-medium | 0.32413 | 0.37210 | +0.04797 | Improved |
| AP-large | 0.51011 | 0.46163 | -0.04848 | Material unplanned regression |
| AR-max | 0.29215 | 0.36598 | +0.07383 | Supporting target passed |
| Mean latency | 10.06 ms | 10.77 ms | +0.71 ms | Guardrail passed; one timing run |
| FPS, batch 1 | 99.4 | 92.9 | -6.5 | Same timing caveat |
| GFLOPs | 6.32 | 14.22 | +7.90 | 2.25x theoretical compute |
| Images at `max_det=300` | 458/548 | 476/548 | +18 images | Saturation rose 3.28 percentage points |

The observed latency increase is much smaller than the GFLOP increase. It is
valid for the recorded runs but should be remeasured before making a deployment
claim because each setting has only one 100-image timing sample.

## TIDE error impacts

Lower dAP is better: it means less AP is recoverable by oracle-fixing that error
category.

| TIDE category | Exp6 | Exp7 | Delta | Result |
|---|---:|---:|---:|---|
| Cls | 14.879 | 13.895 | -0.984 | Improved |
| Loc | 6.252 | 5.086 | -1.166 | Improved; supporting target passed |
| Both | 0.595 | 0.620 | +0.025 | Slightly worse |
| Dupe | 0.555 | 0.631 | +0.076 | Slightly worse |
| Bkg | 2.219 | 3.070 | +0.851 | Worse |
| Miss | 9.288 | 8.655 | -0.633 | Improved, but missed its target by 0.367 |
| Special FalsePos | 12.493 | 15.170 | +2.677 | Worse; guardrail failed by 0.677 |
| Special FalseNeg | 30.890 | 24.643 | -6.247 | Substantially improved |

This is a recall/precision tradeoff. Higher resolution exposed more useful
objects and improved their boxes and class ranking, but it also introduced more
background false-positive opportunity. The simultaneous rise in `max_det`
saturation is consistent with more low-confidence candidates, although it does
not prove that the cap caused any specific failure.

## Preregistered decision check

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Primary AP-small | >=0.12139 | 0.17284 | Pass |
| Supporting AR-max | >=0.30215 | 0.36598 | Pass |
| Supporting TIDE Miss | <=8.288 | 8.655 | Fail |
| Supporting TIDE Loc | <=5.752 | 5.086 | Pass |
| Guardrail mAP50-95 | >=0.21267 | 0.26661 | Pass |
| Guardrail TIDE Cls | <=15.879 | 13.895 | Pass |
| Guardrail special FalsePos | <=14.493 | 15.170 | **Fail** |
| Guardrail mean latency | <=25.15 ms | 10.77 ms | Pass |

The primary target and two of three supporting targets passed, but the written
rule also required every guardrail to pass. Therefore the hypothesis is not
strictly confirmed. The experimental outcome is still positive because the
same checkpoint gained 0.04894 mAP50-95, 0.06145 AP-small, and 0.07383 AR-max,
with improvement in every class.

## Per-class results

| Class | Exp6 AP50-95 | Exp7 AP50-95 | Delta |
|---|---:|---:|---:|
| person | 0.15158 | 0.20608 | +0.05450 |
| bicycle | 0.02930 | 0.05032 | +0.02102 |
| car | 0.51765 | 0.58265 | +0.06500 |
| truck | 0.16199 | 0.19643 | +0.03444 |
| bus | 0.29099 | 0.36615 | +0.07516 |
| motorcycle | 0.15453 | 0.19802 | +0.04349 |

Bicycle remains the weakest class despite its relative improvement. Because
all six classes improved using identical weights, the result strongly supports
insufficient effective input resolution as a real limitation of the 640
operating point. It does not show that training itself improved.

## Root-cause hypotheses after Exp7

| Hypothesis | Supporting evidence | Confidence (0-1) |
|---|---|---:|
| 640 downscaling removed useful small-object evidence | Same weights at 960 gained 0.06145 AP-small and 0.07383 AR-max while Loc fell 1.166 dAP and special FalseNeg fell 6.247 dAP | 0.95 |
| More high-resolution candidates increase false-positive pressure | Bkg rose 0.851 dAP, special FalsePos rose 2.677 dAP, and max-det saturation rose 3.28 percentage points | 0.85 |
| Semantic evidence remains limited for weak classes | Cls improved but remains the largest main TIDE mode at 13.895; bicycle AP remains only 0.05032 | 0.80 |
| 640-trained scale calibration is imperfect at 960 | Exp7 changes inference scale without matching training scale; FP impact and AP-large worsened despite broad AP gains | 0.60 |
| NMS duplicates are the primary limiter | Dupe is only 0.631 dAP | 0.10 |

## Remaining failure modes

The top three mutually exclusive main TIDE modes remain:

1. Wrong-class errors (`Cls` 13.895 dAP).
2. Missed objects (`Miss` 8.655 dAP; special FalseNeg 24.643 is overlapping
   aggregate context, not an additional category).
3. Poor localization (`Loc` 5.086 dAP).

Additional concerns are background false positives (`Bkg` 3.070; special
FalsePos 15.170), the AP-large regression, and prediction saturation at the
fixed 300-detection cap. Duplicate impact remains small, so NMS tuning is not
the first training intervention supported by this result.

## Interpretation

Exp7 answers a narrow question: the existing Exp6 model benefits substantially
from retaining more input pixels at inference. It does not establish that a
960-trained model is better, and it does not optimize confidence thresholds.
The next controlled step is to train the same architecture, taxonomy, sampler,
seed, and epoch budget at 960, then evaluate at the already established 960
protocol. That comparison isolates the effect of matching training resolution
to the higher-resolution inference setting, subject to a smaller batch required
by the 8 GB GPU.
