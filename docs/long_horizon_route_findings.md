# Long-horizon route findings (2026-07-30)

The current official submission scores F1 0.8129. Local radius-2 evaluation of
the corresponding hybrid route is 0.8191 (62,981 TP, 2,814 FP, 25,003 FN).
Track accuracy is already saturated while track completeness is not, but the
missing detections are mostly whole unseen tracks rather than short gaps.

## Experiments rejected

- MTTU-Net author checkpoints were tested with sequence-persistent memory.
  Both zero-shot checkpoints produced no true positives. Polarity inversion
  raised target probabilities by roughly nine orders of magnitude, but still
  produced only 18 TP and 21,666 FP at the most permissive tested threshold.
- Reset-head MTTU-Net adaptation avoided the saturated pretrained output head,
  but learned an almost uniform full-frame response. A further balanced
  positive/negative continuation reduced target ranking and still yielded zero
  top-20 hits.
- Aggressive interpolation and endpoint extrapolation on the high-precision
  256 route reduced F1. At best, 537 added short-gap points contained only 104
  TP and 433 FP. The missing recall cannot be recovered from existing anchors.
- A polarity-invariant hard-negative patch classifier trained on all 640 train
  sequences had only 28.7% candidate coverage on the dark validation domain.
  Training on validation shards 02-04 improved ranking on held-out shard 01
  but not enough to produce useful point precision.
- Dense patch scoring removed the candidate-coverage ceiling, but the target
  median rank was about 3,800. Temporal centering and constant-velocity
  shift-and-stack peaked at F1 0.039.
- A direct nine-frame temporal patch classifier fit its training hard
  negatives but produced zero top-100 hits on a held-out contiguous shard.
- Whole-frame YOLO domain adaptation on validation shards 02-04 produced
  held-out shard-01 point F1 0.0815 (55 TP, 556 FP, 683 FN), improving the
  prior zero result but not enough to merge.
- The corrected scratch P2 detector used 12-pixel centroid boxes and learned
  nonzero box/DFL losses, but held-out point F1 was only 0.0004 (11 TP and
  51,018 FP at the best threshold). The earlier zero-size-label failure was
  fixed, so this is a valid rejection of the single-frame detection route.
- On the first 30 256x256 validation sequences, missed target points had a
  median per-frame fused-response local-maximum rank of about 544. Long-track
  oracle accumulation was separable, but blind dense dynamic programming still
  produced zero hits on a representative wholly missed sequence because
  stronger background paths dominated.

## Domain evidence

For 640x512, nearly all train sequences contain predominantly bright targets,
while all four validation sequences contain predominantly dark targets. The
remaining failure is broader than polarity: even polarity-invariant local
features do not transfer across the sensor/background domain.

## Next viable route

Do not spend more time on threshold sweeps, single-frame detectors, local patch
classifiers, or label-free path search. The next route is end-to-end
spatiotemporal representation learning. The official TDCNet implementation
(AAAI 2026) is being adapted with phase-aligned history in one branch, raw
history in the other, and expanded target boxes. Its full validation point F1,
not its training loss or box mAP, is the merge criterion.
