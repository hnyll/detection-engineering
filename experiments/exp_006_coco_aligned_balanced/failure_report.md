# exp_006_coco_aligned_balanced — Failure Analysis

Exp6 is a custom six-class application task, not official ten-class VisDrone
Task 1. Its metrics use remapped ground truth and must not be presented as a
direct benchmark improvement over Exp1-Exp5. No Exp6 FiftyOne review/export was
performed, so manual outcome counts and condition prevalence are not claimed.
TIDE values are oracle AP impacts, not counts; special FalsePos/FalseNeg values
overlap the main decomposition and must not be added to it.

## Outcomes

| Outcome | Count / impact | Notes |
|---|---:|---|
| Wrong class | Cls dAP 14.879 | Largest individual main TIDE category |
| Missed object | Miss dAP 9.288 | Second-largest main category |
| Poor localization | Loc dAP 6.252 | Third-largest main category |
| Class + localization | Both dAP 0.595 | Neither a pure class nor pure localization error |
| Duplicate | Dupe dAP 0.555 | Small impact; NMS is not a priority |
| Background false positive | Bkg dAP 2.219 | Main TIDE decomposition |
| False positive, aggregate | special dAP 12.493 | Not a raw count; overlaps main errors |
| False negative, aggregate | special dAP 30.890 | Not a raw count; overlaps main errors |

## Per-class results

| Class | AP50-95 | AP50 | Mapping |
|---|---:|---:|---|
| person | 0.15158 | 0.39833 | pedestrian + people |
| bicycle | 0.02930 | 0.06819 | unchanged |
| car | 0.51765 | 0.78191 | car + van |
| truck | 0.16199 | 0.23856 | unchanged and separate from car/bus |
| bus | 0.29099 | 0.41371 | unchanged and separate from car/truck |
| motorcycle | 0.15453 | 0.35960 | motor + tricycle + awning-tricycle |

Bicycle remains the clear weakest class. Cars, trucks, and buses were **not**
combined in Exp6 because COCO retains those three categories, so their mutual
confusion remains a scored classification error.

## Effective object size at 640

The following measurements come from all 38,759 validation labels after
applying the full-image letterbox scale
`min(640 / image_width, 640 / image_height)`. They are recorded in
`artifacts/effective_box_sizes_imgsz640.json`.

| Class | Median model-input box | % with geometric mean <8 px | % with shortest side <4 px |
|---|---:|---:|---:|
| person | 5.6 x 12.3 px | 46.3% | 26.2% |
| bicycle | 9.4 x 10.4 px | 33.4% | 13.7% |
| car | 18.3 x 13.3 px | 18.9% | 6.2% |
| truck | 21.6 x 17.4 px | 12.0% | 2.1% |
| bus | 16.0 x 18.0 px | 15.5% | 2.4% |
| motorcycle | 10.4 x 11.3 px | 30.2% | 10.5% |

This supports a real information bottleneck, but it does not justify deleting
all small labels. Many boxes retain some signal, and removing their labels
while leaving the objects visible would train them as background.

## Confusion-matrix interpretation

Ultralytics' plotted matrix uses confidence 0.25 and matching IoU 0.45. Columns
are true classes and rows are predicted classes. The bottom `background` row is
not a class emitted by YOLO: it contains unmatched ground truths. The rightmost
`background` column contains unmatched predictions/false positives.

At that operating point, approximately 60% of person, 78% of bicycle, 22% of
car, 43% of truck, 41% of bus, and 55% of motorcycle ground truths are
unmatched. Some may have lower-confidence predictions or boxes below the 0.45
IoU match threshold, so these percentages are not equivalent to TIDE `Miss`.
The main visible class confusions are truck to car (about 30% of true trucks)
and bus to car/truck (about 9%/8%).

The fixed evaluation retains predictions down to confidence 0.001 and still
reports Miss dAP 9.288, Loc dAP 6.252, and special FalseNeg dAP 30.890.
Therefore threshold tuning can change the deployment precision/recall tradeoff
but cannot explain away the underlying recall and localization limitations.

## Conditions present in reviewed failures

| Condition | Exp6 reviewed count | Evidence boundary |
|---|---:|---|
| tiny | not manually reviewed | AP-small and effective-size statistics are aggregate evidence, not review counts |
| crowded | not collected | No Exp6 FiftyOne export |
| occluded | not collected | No Exp6 FiftyOne export |
| blur | not collected | No Exp6 FiftyOne export |
| low-light | not collected | No Exp6 FiftyOne export |
| truncation | not collected | No Exp6 FiftyOne export |

## Root-cause hypotheses

| Hypothesis | Supporting evidence | Confidence (0-1) |
|---|---|---:|
| insufficient effective pixels for tiny objects | AP-small is 0.11139 versus 0.32413 medium and 0.51011 large; the median person box is only 5.6 x 12.3 input pixels; Miss and special FalseNeg remain large | 0.90 |
| remaining road-vehicle ambiguity | About 30% of true trucks become cars at the plotted operating point; buses also become cars/trucks. These classes remain separate by design. | 0.75 |
| image-level repeat sampling remains broad | Sampled instances remain highly uneven: car 238,952 versus bicycle 15,083, truck 18,367, and bus 9,505; repeated multi-class images also repeat common objects | 0.70 |
| approximate grouped semantics | `motorcycle` combines motor with two three-wheel categories that COCO does not represent directly; the mapping is application-driven | 0.45 |
| annotation/localization uncertainty | A few pixels of displacement sharply reduce IoU for 5-10 px boxes, but no Exp6 label audit measured annotation noise | 0.40 |
| repeat-factor overfit | Trainer mAP peaked at epoch 70 and declined mildly; `best.pt` protects evaluation and there is no strong divergence | 0.20 |
| NMS suppression is primary | Dupe dAP is only 0.555; current evidence does not support prioritizing NMS | 0.10 |

## Top-3 measured failure modes

1. Remaining wrong-class errors, especially truck/car and bus/car/truck
   separation (`Cls dAP 14.879`).
2. Unmatched and low-confidence small objects, especially bicycle, person, and
   motorcycle (`Miss dAP 9.288`, `AP-small 0.11139`).
3. Localization failures made brittle by very small boxes (`Loc dAP 6.252`).

## Interpretation

Exp6 reaches mAP50-95 0.21767, mAP50 0.37672, mAP75 0.21391, and AR-max
0.29215. Descriptively, mAP50-95 is 0.04502 higher and Cls dAP 8.371 lower than
Exp5, supporting the usefulness of the coarser application taxonomy. These are
different six- and ten-class ground-truth tasks, however, so the deltas are not
an official or controlled benchmark comparison and cannot isolate grouping
from balancing.

The next intervention should test whether retaining more input detail reduces
misses and localization errors while also giving the classifier stronger
evidence. Removing small labels, increasing classification-loss weight, or
tuning NMS is not supported as the first move.
