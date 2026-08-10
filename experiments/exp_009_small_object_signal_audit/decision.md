---
exp: "exp_009_small_object_signal_audit"
hypothesis_confirmed: inconclusive
outcome: inconclusive
next_experiment: ""
---

# exp_009_small_object_signal_audit — Decision

## Status

Abandoned before review. The diagnostic required 120 manual triptych judgments,
which was not worth the time cost for this project. No review CSV was completed,
the hidden answer key was not scored, and no conclusion about native human
recognizability is claimed.

The generated audit artifacts are retained only for reproducibility and can be
deleted later if desired. They are not evidence and must not be cited as a
result.

## Problem

Exp8 remains weak on COCO-small objects, which make up 68.59% of validation
annotations. Before spending another long detector-training run, Exp9 tests
whether those native objects contain recognizable class information and how
much is lost during full-image resizing.

## Possible outcomes

1. **Native and 960 are recognizable; 640 is weak.** Useful signal exists and
   960 preserves most of it. Remaining detector errors point toward search,
   localization, crowding, or the saturated detection cap.
2. **Native is recognizable; both simulated resolutions are weak.** Source
   signal exists but full-image resizing destroys it. Test native-resolution
   tiling/object-context crops next.
3. **Only larger native-small boxes are recognizable.** Define a measured size
   ceiling and decide whether the application should ignore smaller targets or
   use a specialized acquisition/tiling strategy.
4. **Native crops are mostly uncertain or below the signal threshold.** The
   imagery/taxonomy has a real information ceiling. More generic detector
   training is unlikely to solve it; audit labels and application requirements.
5. **Humans confidently disagree with GT.** Label ambiguity/noise is material.
   A targeted annotation audit is needed before changing architecture.

## Interpretation boundaries

This audit is balanced rather than prevalence-weighted, uses one reviewer, and
shows variants together for efficiency. It can guide the next experiment but
cannot establish statistical significance, detector AP, or an official
VisDrone benchmark result.

## Recommendation

Do not spend time on the manual audit. Existing automated evidence is already
sufficient to say that some small-object signal exists: the identical Exp6
checkpoint gained 0.06145 AP-small when inference changed from 640 to 960.
That does not quantify a human/source-image ceiling, but it is adequate for the
current non-production project.

If stronger evidence is needed later, use an automated oracle-crop classifier
or a fixed-weight tiled-inference ablation rather than restarting this manual
review.
