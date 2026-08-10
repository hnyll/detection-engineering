# RT-DETR-L hardware feasibility — RTX 4060 8 GB

## Verdict

`rtdetr-l.pt` is feasible for inference and provisionally feasible for full
fine-tuning on this machine. The conservative 960-pixel training setting is
physical batch 1 with AMP. A 902-target batch-1 step peaked at 3,616 MiB
reserved CUDA memory on an 8,187.5 MiB device. Batch 2 also completed, but its
902-target stress case reached 6,902 MiB and left only about 1,286 MiB for
real-trainer and mosaic variation, so it is not the safe default.

This is a hardware result, not a VisDrone accuracy result. The inference tests
use the stock 80-class COCO checkpoint, and the training tests use a six-class
head with synthetic images and labels.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 |
| Physical VRAM | 8,187.5 MiB |
| PyTorch | 2.9.1+cu128 |
| CUDA reported by PyTorch | 12.8 |
| Ultralytics | 8.4.65 |
| RT-DETR weights SHA-256 | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| Benchmark script SHA-256 | `ec31370032cd4095be6901dc34e590f313fb5d573030e7b17d92b1e3f93ea596` |

## Inference benchmark

The inference benchmark uses batch 1, FP32 PyTorch, five warmup images, and 20
evenly spaced images from the six-class VisDrone validation view. Its timing
includes image decoding, preprocessing, model execution, and Ultralytics
postprocessing, but excludes initial model loading and kernel warmup.

| Model | Input | Params | GFLOPs | Mean | p50 | p95 | FPS | Peak reserved VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 640 | 2.624 M | 6.61 | 10.98 ms | 10.52 ms | 10.99 ms | 91.1 | 68 MiB |
| RT-DETR-L | 640 | 32.970 M | 108.34 | 31.57 ms | 31.37 ms | 35.79 ms | 31.7 | 284 MiB |
| YOLO11n | 960 | 2.624 M | 14.88 | 12.24 ms | 12.13 ms | 13.12 ms | 81.7 | 90 MiB |
| RT-DETR-L | 960 | 32.970 M | 239.47 | 52.81 ms | 51.86 ms | 59.13 ms | 18.9 | 408 MiB |

At 960, RT-DETR-L is about 4.32 times slower than YOLO11n in this matched
measurement. It is below a 30 FPS real-time target at 960, while 640 narrowly
exceeds 30 FPS. TensorRT or FP16 inference could change the deployment result,
but neither was measured here.

RT-DETR is NMS-free and returns a top-300 query set; its `iou` prediction
argument is not equivalent to YOLO NMS IoU. It also uses scale-fill square
preprocessing rather than YOLO's aspect-preserving letterbox. Consequently,
the comparison is a model-system cost comparison, not a pure architecture
ablation.

## Training-memory benchmark

The training smoke instantiates the real six-class RT-DETR-L model, transfers
926/941 compatible tensors from the COCO checkpoint, and executes AMP forward,
RT-DETR loss, backward, AdamW, and EMA. The synthetic labels exercise the
denoising-query memory path. The second step excludes first-use kernel/startup
cost and is used only for a rough throughput projection.

| Input | Batch | Targets/image | Status | Steady step | Peak allocated | Peak reserved | Process RSS |
|---|---:|---:|---|---:|---:|---:|---:|
| 640 | 1 | 100 | pass | not measured | 1,291 MiB | 1,364 MiB | not recorded |
| 960 | 1 | 100 | pass | 207 ms | 2,400 MiB | 2,550 MiB | 2,750 MiB |
| 960 | 1 | 902 | pass | startup step only | 3,359 MiB | 3,616 MiB | 2,754 MiB |
| 960 | 2 | 100 | pass | 275 ms | 3,352 MiB | 3,594 MiB | 2,734 MiB |
| 960 | 2 | 902 | pass | 505 ms | 6,293 MiB | 6,902 MiB | 2,757 MiB |

The repeated training manifest has 8,716 entries. Native label counts have a
median of 44, p95 of 137, p99 of 214, and maximum of 902. At 207 ms per typical
batch-1 step, 100 epochs have a 50.2-hour model-step lower bound. Because this
estimate comes from only one post-startup synthetic step and omits real data
loading, augmentation, validation, and checkpointing, use a conservative
planning range of roughly 60–80 hours. A measured real-data smoke should
replace that estimate before committing to the whole run.

The synthetic smoke does not prove that every full-trainer code path will fit.
A 100–200-batch real-data smoke with mosaic, checkpoint save/reload, and peak
memory capture should precede a full run. Batch 2 may be reconsidered only if
that smoke remains stable with useful VRAM headroom.

## Recommended full experiment settings

```yaml
train:
  model: rtdetr-l.pt
  imgsz: 960
  epochs: 100
  batch: 1
  nbs: 64
  amp: true
  deterministic: false
  workers: 2
  cache: false
```

Use the same six-class balanced training manifest, seed 17, and 960 evaluation
dataset as the current control, but do not carry over `cls_pw`: it is a
YOLO-specific loss treatment and is not part of RT-DETR's native loss.

Before claiming a fair accuracy comparison, the experiment harness must record
RT-DETR's scale-fill preprocessing, NMS-free 300-query postprocessing, and the
fact that NMS IoU is unused as part of the evaluation-pipeline identity. Start
from the stock checkpoint rather than continuing a YOLO checkpoint.

## Reproduction

The isolated runner is `scripts/benchmark_rtdetr_hardware.py`. Example:

```bash
.venv/bin/python scripts/benchmark_rtdetr_hardware.py \
  --mode inference --model rtdetr-l.pt --imgsz 960 --batch 1 \
  --samples 20 --warmup 5

.venv/bin/python scripts/benchmark_rtdetr_hardware.py \
  --mode train-step --model rtdetr-l.pt --imgsz 960 --batch 1 \
  --objects 902 --steps 2
```

The JSON files beside this report contain the raw measurements and model,
software, GPU, and checkpoint identities.
