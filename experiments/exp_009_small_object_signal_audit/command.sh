#!/usr/bin/env bash

# 1. Build deterministic blinded crops, answer key, and local browser reviewer.
uv run python scripts/small_object_signal_audit.py build \
  --experiment exp_009_small_object_signal_audit

# 2. Open artifacts/signal_audit/review.html in a browser. Complete as many
# samples as practical (target: >=90 total and >=15 per class), then use the
# page's Export CSV button. Save the download as:
# experiments/exp_009_small_object_signal_audit/artifacts/signal_audit/review_completed.csv

# Optional local server if the browser restricts file:// JavaScript:
# cd experiments/exp_009_small_object_signal_audit/artifacts/signal_audit
# python3 -m http.server 8000
# Then open http://localhost:8000/review.html

# 3. Score only after review is complete; this is when answers.csv is joined.
uv run python scripts/small_object_signal_audit.py score \
  --experiment exp_009_small_object_signal_audit \
  --review experiments/exp_009_small_object_signal_audit/artifacts/signal_audit/review_completed.csv
