# detection-engineering

A competency-based object detection engineering harness: **YOLO11 on VisDrone-DET**,
instrumented with Weights & Biases, FiftyOne, TIDE error analysis, and
ONNX/TensorRT/Core ML deployment benchmarks. Built to develop engineering judgment
through controlled experiments — not benchmark chasing.

The methodology lives in [`docs/harness_spec.md`](docs/harness_spec.md): six
competencies (Train, Diagnose, Improve, Compare, Deploy, Explain), each with
mechanical exit criteria checked by `make gates`. Every experiment changes one
causal variable, pre-registers its prediction, and ends with a decision.

## Quickstart

```bash
make setup     # installs uv if needed, syncs .venv from uv.lock (exact, reproducible)
make data      # links the local VisDrone copy into data/
make doctor    # verifies CUDA, imports, FiftyOne DB, W&B mode
make phase0    # COCO128 plumbing smoke test: train -> resume -> eval -> TIDE ->
               # FiftyOne -> ONNX export + parity -> gates snapshot
```

Then real work:

```bash
make new NAME=baseline_640              # scaffold experiments/exp_001_baseline_640
# edit hypothesis.md (prediction BEFORE training) and config.yaml
make train EXP=exp_001_baseline_640 ARGS="--seeds 17,42,1337"
make eval  EXP=exp_001_baseline_640
make tide  EXP=exp_001_baseline_640
make review EXP=exp_001_baseline_640 ARGS="--launch"   # FiftyOne at localhost:5151
make compare EXPS="exp_001_baseline_640 exp_002_res1024"
make gates                                              # where do I stand?
```

## Experiment anatomy

```
experiments/exp_001_baseline_640/
├── hypothesis.md       # question + pre-registered prediction + the ONE causal variable
├── config.yaml         # meta (dataset, seeds, parent) + ultralytics train args
├── command.sh          # exact reproduction command, git SHA, package versions
├── metrics.json        # fixed-protocol COCOeval: mAP, per-class, S/M/L, latency, params
├── tide_report.json    # error taxonomy: Cls / Loc / Both / Dupe / Bkg / Miss dAP
├── failure_report.md   # outcomes vs conditions vs hypotheses (with confidence)
├── decision.md         # decision playbook; frontmatter feeds the gates checker
├── seeds/s17/ ...      # per-seed runs (weights gitignored)
└── artifacts/          # plots, crops, compare tables (heavy files gitignored)
```

## Competency gates

`make gates` prints ✓/✗ against the spec's exit criteria — e.g. Train needs a
reproducible baseline with a proven checkpoint resume and a W&B run; Improve needs
≥3 controlled hypotheses with at least one documented negative result; Compare
needs 3-seed variance and refuses cross-protocol comparisons.

## Experiment ledger

The repository records both useful and negative results. Values below are
fixed-protocol validation results; rows with different taxonomies or inference
pipelines are not leaderboard-equivalent.

| Experiment | Isolated treatment | Task / protocol | mAP50-95 | Decision |
|---|---|---|---:|---|
| Exp1 | YOLO11n baseline at 640 | VisDrone 10-class, 3 seeds | 0.17049 ± 0.00074 | reference baseline |
| Exp2 | train at 1024, evaluate at 640 | VisDrone 10-class, 3 seeds | 0.17290 ± 0.00188 | reject: tiny AP regressed |
| Exp3 | merge into 3 coarse groups | custom 3-class task | 0.27391 | useful taxonomy diagnostic; not Task 1 |
| Exp4 | add a P2 detection head | VisDrone 10-class | 0.16470 | negative |
| Exp5 | repeat-factor image sampling | VisDrone 10-class | 0.17265 | negative |
| Exp6 | COCO-aligned 6 classes + balancing | custom 6-class task | 0.21767 | application baseline |
| Exp7 | same Exp6 checkpoint, infer at 960 | custom 6-class / 960 | 0.26661 | positive with FP caveat |
| Exp8 | train and evaluate at 960 | custom 6-class / 960 | 0.28847 | positive |
| Exp10 | inverse-frequency class loss | custom 6-class / 960 | 0.29311 | inconclusive; Cls dAP unchanged |
| Exp11 | native 640 tiles only | custom tiled pipeline | 0.31128 | tiny-object mechanism positive; deployment negative |
| Exp12 | full-frame + tiled fusion | custom hybrid pipeline | 0.32360 | best accuracy; FP/duplicate pressure remains |
| Exp13 | ConvNeXt-T + FPN Faster R-CNN | custom 6-class / 960 | pending | hardware smoke passed; training interrupted |

Exp9 was an analysis-only small-object signal audit and was stopped before the
manual review. Full hypotheses, protocol identities, TIDE reports, guardrails,
and caveats live in each experiment directory.

## Hardware honesty

Everything is tuned for one RTX 4060 (8 GB) on WSL2 — n/s models, AMP, explicit
batch sizes, dataloader workers capped, RSS logged per run. See
[`docs/machine_notes.md`](docs/machine_notes.md). Deployment targets:
ONNX → TensorRT (FP16/INT8) on the 4060, Core ML for iPhone 13 Pro
([`docs/deploy_coreml.md`](docs/deploy_coreml.md)).
