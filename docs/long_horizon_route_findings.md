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
- The official TDCNet box head remained ineffective after CSIG adaptation. A
  full-resolution heatmap replacement also failed: epoch 2 reached only about
  `0.002` F1 on the fixed 256 probe, so neither output formulation justified
  longer scratch training.
- A MoPKL-derived detector initialized from the released visual backbone
  reached F1 `0.2834` on the first 30 256 sequences after three epochs. This
  confirms non-random learning but also confirms that the single-scale
  stride-8 YOLOX head cannot approach the current point detector without the
  unavailable task checkpoint.
- FeedbackSTS-Det was adapted with a torchvision deformable-convolution
  replacement. The original loss collapsed to background; a positive-crop,
  dilated-mask, separately normalized focal variant reached only F1 `0.1230`
  on the same 30-sequence probe after three epochs.
- Loddis was inspected but not trained blindly. Its released forward pass
  discards all but the last input frame and uses a single stride-8 YOLOX head;
  its contribution is domain-adversarial object/background factorization.
  That factorization addresses domain shift, but the released detector repeats
  the spatial and temporal limitations already measured in MoPKL-lite.
- Pretrained RAFT-small was used to warp neighboring frames at lags 2, 4 and
  8. On the first 30 256 sequences, flow-compensated motion anomalies reached
  standalone F1 `0.3889`. Of 2,196 points missed by the current baseline only
  461 appeared among the top 20 anomaly peaks. Adding two peaks per frame
  reduced F1 from `0.8023` to `0.7179`, so direct optical-flow complementation
  was rejected.
- HDNet was converted to a three-channel temporal-frequency model using the
  current frame and forward/backward differences. Exact shape matching reused
  2.84M of its 3.68M parameters from the trained MSHNet. After three bounded
  epochs it reached only F1 `0.1366` on the first 30 256 sequences.
- A DeepPro weak-track curriculum attenuated target contrast, reversed target
  polarity and sometimes inverted the whole frame while retaining the strong
  checkpoint initialization. Its two-epoch stage reached only about F1
  `0.703` on the first 30 sequences versus the unchanged checkpoint route's
  `0.802`, losing true positives instead of expanding recall.

## Domain evidence

For 640x512, nearly all train sequences contain predominantly bright targets,
while all four validation sequences contain predominantly dark targets. The
remaining failure is broader than polarity: even polarity-invariant local
features do not transfer across the sensor/background domain.

## Next viable route

Do not spend more time on threshold sweeps, direct flow residuals, single-frame
detectors, local patch classifiers, short scratch training, or label-free path
search. Every such family now has a measured recall or precision ceiling.

The remaining high-upside route is task-specific pretrained spatiotemporal
representation, followed by CSIG calibration rather than head replacement.
The official MoPKL repository publishes strong DAUB-R and IRDST-H checkpoints,
but the files are hosted as large Baidu Netdisk downloads that require an
authenticated native-client session; the anonymous web API returns only the
encrypted client payload. Once either checkpoint is placed locally, run
zero-shot point-F1 and missed-track coverage first, then fine-tune only if the
checkpoint demonstrates genuine complementarity. S²CPNet and decoupled motion
representation learning are also conceptually aligned with the measured domain
gap, but no official executable weights were found, so implementing them from
paper descriptions is lower priority than acquiring the released MoPKL model.
