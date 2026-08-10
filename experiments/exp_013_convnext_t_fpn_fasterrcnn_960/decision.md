---
exp: "exp_013_convnext_t_fpn_fasterrcnn_960"
hypothesis_confirmed: null
outcome: pending
next_experiment: null
---

# exp_013_convnext_t_fpn_fasterrcnn_960 — Decision

## Hardware smoke

The five-batch real-data seed-17 smoke passed. It deliberately used the five
densest unique samples first (902, 512, 462, 431, and 370 targets). The complete
forward/loss/backward/AdamW steps remained finite and peaked at 5,018 MiB
reserved VRAM on the 8,188 MiB RTX 4060. After the one-time 37.3-second first
step, the remaining steps took 105–165 ms (110 ms median).

At the median measured step time, 24 epochs over the 8,716-entry manifest have
about 6.4 hours of pure training-step work. Validation, checkpointing, data
loading, and long-run variation are not included; plan roughly 7–10 hours until
the first complete run measures the true wall time.

The smoke proves short-run hardware feasibility, not convergence, evaluation
correctness, or accuracy. Training, fixed evaluation, and TIDE remain pending.
