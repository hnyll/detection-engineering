# exp_009_small_object_signal_audit — Signal Audit

**Status: abandoned before manual review.** No completed labels or scored
results exist. The blank tables below are retained as the preregistered plan,
not as evidence.

Complete after exporting the browser review CSV and running the scoring
command. This is a GT-blinded human crop audit, not detector evaluation.

## Audit completion

| Check | Expected | Observed |
|---|---:|---:|
| GT fingerprint | `e1142868d4dc2e9d` | `e1142868d4dc2e9d` |
| Requested/generated samples | 120 | 120 |
| Unique source images | preferably 120 | 120; no reuse relaxation |
| Per-class/stratum allocation | 20 each; 7/7/6 | exact for all six classes |
| Reviewed samples | >=90 total and >=15/class | |
| Sampling seed | 17 | 17 |
| Context scale/minimum/maximum | 4.0 / 64 / 256 px | 4.0 / 64 / 256 px |
| Hidden answer-key SHA-256 | frozen before review | `a6c60c97dd39820708258a8fd9277a3cd6d3c828bb66b1039fb46c999874c64e` |
| Review-template SHA-256 | frozen before review | `ca9ed5213c03f12b7670c10af67b7efe9ea6b22754ea22927cf8ae388e31e1b5` |

## Aggregate recognizability

| Variant | Reviewed | Accuracy | Macro accuracy | Coverage | Accuracy when answered | Uncertain/not visible |
|---|---:|---:|---:|---:|---:|---:|
| simulated 640 | | | | | | |
| simulated 960 | | | | | | |
| native | | | | | | |

## Preregistered checks

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Native macro accuracy | >=0.50 | | |
| Native coverage | >=0.70 | | |
| Native minus simulated-640 accuracy | >=0.15 | | |
| Simulated-960 minus simulated-640 accuracy | >=0.08 | | |
| Reviewed per class | >=15 | | |

## Results by native shortest-side bin

| Bin | Samples | 640 accuracy | 960 accuracy | Native accuracy | Native uncertainty | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| micro (`<8 px`) | | | | | | |
| tiny (`8-<16 px`) | | | | | | |
| small (`>=16 px`, area `<32^2`) | | | | | | |

## Results by class

| Class | Samples | 640 accuracy | 960 accuracy | Native accuracy | Common confusion/notes |
|---|---:|---:|---:|---:|---|
| person | | | | | |
| bicycle | | | | | |
| car | | | | | |
| truck | | | | | |
| bus | | | | | |
| motorcycle | | | | | |

## Observable conditions

Summarize review notes without turning them into unmeasured causal claims.

| Condition | Reviewed examples | Notes |
|---|---:|---|
| not visible even natively | | |
| class ambiguous | | |
| identifiable only with context | | |
| likely questionable GT | | |
| heavy occlusion/truncation | | |

## Decision prompts

- Is native class signal comfortably above the balanced 16.7% chance level?
- Does 960 retain materially more signal than 640?
- Is native materially better than 960, supporting tiles/native crops?
- Are failures concentrated below a particular shortest-side threshold?
- Are low results caused by `uncertain`/`not_visible`, or confident disagreement
  with the GT taxonomy?
- If native crops are recognizable, should the next detector experiment target
  localization/candidate saturation rather than more semantic training?

## Conclusion

Fill only after scoring. Do not infer population prevalence from this balanced
sample or claim that a single human review establishes model learnability.
