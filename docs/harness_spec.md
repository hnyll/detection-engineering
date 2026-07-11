# Mastering Object Detection Engineering Harness

This harness is organized around competencies with exit criteria instead
of time. It is designed to build engineering judgment, produce strong
resume projects, and prepare for Applied ML / Computer Vision
interviews. Workflows, schemas, and competency gates make it executable
rather than descriptive. \## Core Competencies - 1. Train Exit criteria:
reproducible baseline, config committed, checkpoint resume works, W&B
synced, second clean run succeeds. - 2. Diagnose Exit criteria:
class-wise/APs metrics, TIDE report, FiftyOne review of ≥100 failures,
top-3 failure modes identified with evidence. - 3. Improve Exit
criteria: ≥3 controlled hypotheses tested independently, at least one
negative result documented, conclusions tied to evidence. - 4. Compare
Exit criteria: same evaluation protocol, latency/FPS/FLOPs/params
recorded, variance measured with 3 seeds for important comparisons. - 5.
Deploy Exit criteria: ONNX and CoreML/TensorRT exports verified, latency
profiled, numerical drift documented. - 6. Explain Exit criteria:
README, resume bullet, STAR story, 2-minute walkthrough, answers to
follow-up questions. \## Stack and Tooling - Framework: Ultralytics
(YOLO11/YOLOv8) first --- fastest path to reproducible baselines.
Graduate to reading and reimplementing papers (RT-DETR, DINO, assignment
and loss papers) once the loop is second nature. - Experiment tracking:
Weights & Biases --- auto-logs every run's config, metric curves, and
checkpoints so ablations are comparable side by side. Ultralytics
integrates natively. - Failure analysis: FiftyOne + TIDE. Do not build
custom tooling. - Hardware budget: RTX 4060 8 GB --- n/s model sizes,
AMP on, gradient accumulation for high-resolution runs. \## Dataset
Plan - Phase 0 --- COCO128 smoke test (1--2 days): prove the plumbing
only --- train, eval, W&B logging, FiftyOne loading, export. Draw no
modeling conclusions from it. - Phase 1 --- VisDrone-DET (the real
project): 6,471 train images, 10 classes, dense tiny objects, occlusion,
class imbalance. Expect roughly 20 mAP at 640 px from a stock baseline
(SOTA ≈ 40); low numbers are the point --- rich failure modes. -
Dataset-driven ablations: input resolution 640 → 1024 → 1280, SAHI
tiling, model size vs resolution tradeoff under the 8 GB budget. \##
Experiment Directory Schema - experiments/exp_XXX_name/ - ├──
hypothesis.md - ├── config.yaml - ├── command.sh - ├── metrics.json -
├── tide_report.json - ├── failure_report.md - ├── decision.md - └──
artifacts/ (plots, predictions, crops) \## Experiment Workflow -
Question - Prediction (written before training) - Background reading -
Baseline - One causal variable changed - Necessary implementation
adjustments (e.g. batch size due to VRAM) - Training - Metrics - TIDE +
FiftyOne review - Root-cause hypotheses - Decision - Next experiment \##
Failure Analysis Schema - Separate observable conditions from outcomes
and hypotheses. - Outcome: TP / FP / FN / Duplicate / Wrong Class / Poor
Localization. - Conditions: tiny, crowded, occluded, blur, low-light,
truncation. - Hypotheses: assignment failure, feature resolution, NMS
suppression, label noise, domain shift. - Store confidence level for
each hypothesis. \## Decision Playbook - Problem - Symptoms (metrics) -
Possible causes ranked - Evidence from previous experiments - Solution
families (Data / Architecture / Loss / Assignment / Training /
Deployment) - Tradeoffs - Experiments to validate - Recommendation \##
Statistical Discipline - Measure baseline variance with three random
seeds. - Treat improvements smaller than observed variance as
inconclusive. - Do not compare results across different evaluation
protocols. \## Interview Companion - Generate one resume bullet using
Action + Scale + Result + Engineering Judgment. - Generate one STAR
story. - Generate one 2-minute project explanation. - Generate five
likely interviewer questions and evidence-based answers. - Every claim
must reference experiment artifacts. \## Deployment Targets -
Workstation: ONNX → TensorRT on the RTX 4060; benchmark FP16 vs INT8
latency against accuracy. - Edge: Core ML on iPhone 13 Pro (A15 Neural
Engine). Export FP16 via Ultralytics (format=coreml, nms=True); profile
with Xcode Core ML performance reports. - Compare backends: quantization
tradeoffs, unsupported ops, and accuracy drift between exports. \##
Guiding Rules - Change one causal factor at a time. - Document
compensating implementation changes separately. - Negative results are
valuable. - Every experiment must end with a decision. - Optimize for
engineering understanding, not benchmark chasing.
