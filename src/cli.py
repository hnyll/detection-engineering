"""Harness CLI. Every competency workflow is one subcommand (see Makefile aliases)."""

from __future__ import annotations

import argparse


def _seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="scaffold a new experiment from templates")
    sp.add_argument("name")

    sp = sub.add_parser("train", help="train (per-seed) with machine-safe defaults")
    sp.add_argument("exp")
    sp.add_argument("--seeds", type=_seeds, default=None)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--interrupt-after", type=int, default=None,
                    help="raise KeyboardInterrupt after N epochs (resume-proof test)")

    sp = sub.add_parser("eval", help="fixed-protocol evaluation -> metrics.json")
    sp.add_argument("exp")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--weights", default=None)

    sp = sub.add_parser("tide", help="TIDE error taxonomy -> tide_report.json")
    sp.add_argument("exp")
    sp.add_argument("--seed", type=int, default=None)

    sp = sub.add_parser("review", help="FiftyOne failure review")
    sp.add_argument("exp")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--launch", action="store_true", help="open the app on :5151")
    sp.add_argument("--export-stats", action="store_true",
                    help="write artifacts/fiftyone_review.json (gate evidence)")

    sp = sub.add_parser("compare", help="compare experiments under the same protocol")
    sp.add_argument("exps", nargs="+")

    sp = sub.add_parser("export", help="export weights for deployment")
    sp.add_argument("exp")
    sp.add_argument("--format", default="onnx", choices=["onnx", "engine", "coreml"])
    sp.add_argument("--half", action="store_true")
    sp.add_argument("--int8", action="store_true")
    sp.add_argument("--seed", type=int, default=None)

    sp = sub.add_parser("parity", help="pt vs ONNX numerical drift report")
    sp.add_argument("exp")
    sp.add_argument("--seed", type=int, default=None)

    sp = sub.add_parser("bench", help="backend latency benchmark")
    sp.add_argument("exp")
    sp.add_argument("--backend", required=True, choices=["pt", "onnx", "trt-fp16", "trt-int8"])
    sp.add_argument("--seed", type=int, default=None)

    sub.add_parser("gates", help="competency exit-criteria report")

    sp = sub.add_parser("interview", help="generate interview companion skeletons")
    sp.add_argument("--exp", default=None)

    sub.add_parser("phase0", help="COCO128 end-to-end plumbing smoke test")

    sp = sub.add_parser("publish", help="git/GitHub publishing helpers")
    sp.add_argument("--init", action="store_true")
    sp.add_argument("--create-remote", action="store_true")
    sp.add_argument("--push", action="store_true")
    sp.add_argument("-m", "--message", default=None)

    sub.add_parser("doctor", help="environment health checks")

    a = p.parse_args()

    if a.cmd == "new":
        from .scaffold import new_experiment
        new_experiment(a.name)
    elif a.cmd == "train":
        from .train import train
        train(a.exp, seeds=a.seeds, resume=a.resume,
              interrupt_after=a.interrupt_after, epochs=a.epochs)
    elif a.cmd == "eval":
        from .evaluate import evaluate
        evaluate(a.exp, seed=a.seed, weights=a.weights)
    elif a.cmd == "tide":
        from .tide_wrap import run_tide
        run_tide(a.exp, seed=a.seed)
    elif a.cmd == "review":
        from .fo_review import review
        review(a.exp, seed=a.seed, launch=a.launch, export_stats=a.export_stats)
    elif a.cmd == "compare":
        from .compare import compare
        compare(a.exps)
    elif a.cmd == "export":
        from .export import export_model
        export_model(a.exp, fmt=a.format, half=a.half, int8=a.int8, seed=a.seed)
    elif a.cmd == "parity":
        from .export import parity
        parity(a.exp, seed=a.seed)
    elif a.cmd == "bench":
        from .export import bench
        bench(a.exp, backend=a.backend, seed=a.seed)
    elif a.cmd == "gates":
        from .gates import report
        report()
    elif a.cmd == "interview":
        from .interview_gen import generate
        generate(exp_ref=a.exp)
    elif a.cmd == "phase0":
        from .phase0 import run_phase0
        run_phase0()
    elif a.cmd == "publish":
        from .publish import publish
        publish(init=a.init, create_remote=a.create_remote, push=a.push, message=a.message)
    elif a.cmd == "doctor":
        from .doctor import doctor
        doctor()


if __name__ == "__main__":
    main()
