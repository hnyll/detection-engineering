"""Environment health checks. Hard failures (torch/CUDA/ultralytics/pycocotools)
exit non-zero; optional layers (tidecv, tensorrt, coreml) only warn."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .paths import DATA, ROOT

OK, WARN, FAIL = "✓", "!", "✗"


def _row(mark: str, label: str, detail: str = "") -> None:
    print(f"  {mark} {label}" + (f"  [{detail}]" if detail else ""))


def doctor() -> None:
    hard_fail = False
    print("doctor:")

    _row(OK, "python", sys.version.split()[0])

    try:
        import torch  # must precede onnxruntime (bundled cuDNN resolution)
        cuda = torch.cuda.is_available()
        _row(OK if cuda else FAIL, "torch + CUDA",
             f"{torch.__version__}, {torch.cuda.get_device_name(0) if cuda else 'NO GPU'}")
        hard_fail |= not cuda
    except Exception as e:
        _row(FAIL, "torch", str(e)); hard_fail = True

    try:
        import ultralytics
        _row(OK, "ultralytics", ultralytics.__version__)
    except Exception as e:
        _row(FAIL, "ultralytics", str(e)); hard_fail = True

    try:
        import numpy
        import pycocotools  # noqa: F401
        _row(OK, "numpy + pycocotools", f"numpy {numpy.__version__}")
    except Exception as e:
        _row(FAIL, "numpy/pycocotools", str(e)); hard_fail = True

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu = "CUDAExecutionProvider" in providers
        _row(OK if gpu else WARN, "onnxruntime",
             f"{ort.__version__}, CUDA {'yes' if gpu else 'NO — parity/bench on CPU'}")
    except Exception as e:
        _row(WARN, "onnxruntime", str(e))

    try:
        from .train import _wandb_mode
        import wandb
        mode = _wandb_mode()
        mark = OK if mode == "online" else WARN
        _row(mark, "wandb", f"{wandb.__version__}, mode={mode}"
             + ("" if mode == "online" else " — run `uv run wandb login` for cloud sync"))
    except Exception as e:
        _row(WARN, "wandb", str(e))

    try:
        import fiftyone as fo
        n = len(fo.list_datasets())  # boots the bundled mongod
        _row(OK, "fiftyone (mongod boots)", f"{fo.__version__}, {n} dataset(s)")
    except Exception as e:
        _row(WARN, "fiftyone", f"{e} — try FIFTYONE_DATABASE_URI (docs/machine_notes.md)")

    try:
        from .tide_wrap import _numpy_shim
        _numpy_shim()
        import tidecv  # noqa: F401
        _row(OK, "tidecv", "primary TIDE path available")
    except Exception as e:
        _row(WARN, "tidecv", f"{e} — greedy-count fallback will be used")

    for opt in ("sahi", "coremltools", "tensorrt"):
        try:
            mod = __import__(opt)
            _row(OK, opt, getattr(mod, "__version__", ""))
        except Exception as e:
            _row(WARN, opt, f"unavailable ({type(e).__name__})")

    vis = DATA / "visdrone"
    if vis.exists():
        n_val = len(list((vis / "images" / "val").glob("*.jpg")))
        _row(OK if n_val == 548 else WARN, "visdrone data", f"{n_val} val images (expect 548)")
    else:
        _row(WARN, "visdrone data", "missing — run `make data`")

    gh = shutil.which("gh")
    if gh:
        v = subprocess.run([gh, "--version"], capture_output=True, text=True).stdout.split("\n")[0]
        _row(OK, "gh CLI", v)
    else:
        _row(WARN, "gh CLI", "not found — publish --create-remote unavailable")

    _row(OK if (ROOT / ".git").exists() else WARN, "git repo", "")

    if hard_fail:
        print("doctor: HARD FAILURES — fix before training")
        raise SystemExit(1)
    print("doctor: core environment healthy")
