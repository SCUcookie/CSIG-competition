# Competition research

Access date: 2026-07-23. Primary sources are the current V3 site (`https://jinsight-cup.github.io/JinSight-ChallengeV3/`), `prompt.md`, `参赛须知与提交结果说明(3).pdf`, and `Codabench 比赛操作流程(3).pdf`. The current V3 statement is authoritative over historical material.

## Task and current rules

Track 1 is infrared satellite video moving-target detection, with centroid detection as the main task and tracking as an additional task. The V3 site describes 1,400 sequences split 1,000/200/200 train/validation/test, comprising IRAir, IRSatVideo-LEO, IRSatVideo-GEO and measured satellite thermal-infrared video. Train/validation labels are released according to the current materials; test labels are not public. Exact schedule dates, submission-count limits, official matching tolerances and detailed per-source resolution distributions must be checked against the current PDFs/site at submission time and are not invented here.

The PDF describes one same-named TXT per sequence, one-based five-digit frame numbers, empty-frame lines, detection mode ID 0, and distinct IDs in tracking mode. The ZIP root must contain TXT files directly, with no subdirectories. This repository validates these properties offline; its metrics are not the official scorer.

## Metrics and workflow

The documented metric vocabulary includes Recall, Precision, F1, trajectory completeness and trajectory accuracy, with a final score combining the official stage-specific terms. The exact formula, tolerance and two-stage submission limits belong to the current PDF and should be confirmed before upload. No Codabench upload or scoring container was run.

## DeepPro and history

The pinned DeepPro reference is commit `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28` from `https://github.com/TinaLRJ/DeepPro`. Its older README reports a SatVideoIRSDT split of 1001/202/200, 1,403 sequences, and a temporal-window baseline. Those figures are historical and are not substituted for V3's 1,400. The pinned source was not copied into this repository and no weights were downloaded or run. Method details and historical baseline numbers should be quoted only from that pinned README and the dataset paper, not inferred from names.

| Source/version | sequences/split | data naming/composition | use | decision |
|---|---:|---|---|---|
| Current V3 site | 1400; 1000/200/200 | four current categories above | current rules | authoritative |
| Pinned DeepPro README | 1403; 1001/202/200 | older SatVideoIRSDT naming | historical baseline/tool | report as historical |
| Dataset paper | historical dataset and competition context | paper-era names and baselines | research evidence | do not overwrite V3 |

## Coordinate-order risk

Code in the fixed DeepPro tree's `tools_forSatVideoIRSTD/seg2centroid_txt.py` first obtains OpenCV `(cx, cy)` but reportedly writes `(row, col)`, suggesting `(y, x)`. The current submission PDF says “horizontal coordinate, vertical coordinate” and does not settle the online scorer's interpretation. This project fixes internal `(x=column, y=row)`, defaults to `xy`, and exposes `--coordinate-order xy|yx`. The same asymmetric example is internal `(2.5,9.5)`, output as `2.5 9.5` in `xy` or `9.5 2.5` in `yx`. Before a real submission, confirm using an official sample or scorer feedback; do not infer it from the old script alone.

## Open questions

Confirm exact schedule dates, submission limits, official point matching and final-score formula; confirm data directory/label pairing and resolution distribution after extraction; confirm coordinate order with an official sample; obtain the pinned DeepPro license status and compatible weights before optional inference.
