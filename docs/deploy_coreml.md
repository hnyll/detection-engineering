# Core ML deployment — manual steps (requires a Mac)

Conversion runs on Linux (`make export EXP=... ARGS="--format coreml"` → `.mlpackage`
via coremltools, FP16, `nms=True`). Profiling and accuracy validation cannot run on
Linux — complete this checklist on a Mac with Xcode and record results below.

Target: iPhone 13 Pro (A15 Neural Engine).

## Checklist
- [ ] Copy `experiments/<exp>/exports/*.mlpackage` to the Mac.
- [ ] Open in Xcode → Core ML performance report on a connected iPhone 13 Pro
      (Compute Unit: All / Neural Engine).
- [ ] Record: median prediction latency (ms), compute-unit dispatch
      (% Neural Engine / GPU / CPU), any ops that fell back to CPU.
- [ ] Run ≥20 val images through the Core ML model (Vision or coremltools on Mac);
      compare boxes/confidences against the ONNX parity baseline; note drift.
- [ ] Paste results into the experiment's `decision.md` and commit.

## Results
| Field | Value |
|---|---|
| Date / device | |
| Median latency (ms) | |
| Neural Engine dispatch % | |
| CPU-fallback ops | |
| Accuracy drift vs ONNX | |
