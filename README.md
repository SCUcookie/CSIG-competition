# JinSight Track 1 baseline

CPU-only engineering baseline for the CSIG2026 Track 1 workflow. Install with `python -m pip install -e .` and run:

```bash
python -m jinsight_track1.cli inspect-data /path/to/data --max-sequences 2 --max-frames 10
python -m jinsight_track1.cli smoke
python -m jinsight_track1.cli infer /path/to/data predictions --max-sequences 2 --max-frames 20
python -m jinsight_track1.cli infer /path/to/data predictions --track --coordinate-order yx
python -m jinsight_track1.cli package predictions submission.zip --expected 2
python -m jinsight_track1.cli estimate-hardware
python -m jinsight_track1.cli train /path/to/train work_dirs/baseline.json --max-frames 2000
python -m jinsight_track1.cli baseline-infer /path/to/val work_dirs/baseline.json submissions/val_baseline
```

Internal coordinates are always `(x=column, y=row)`. Default submission order is `xy`; `yx` swaps only at serialization/parsing boundaries. For example, internal `(2.5, 9.5)` is `ID 2.5 9.5` in `xy` and `ID 9.5 2.5` in `yx`. Detection mode is the default and uses ID `0`; tracking requires `--track` and uses sequence-local IDs starting at 1.

The fake detector and `smoke` command are deterministic and do not load weights. The included `train` command is a deliberately weak, CPU-only global-intensity threshold baseline; it is useful for validating the full workflow, not for claiming competitive accuracy. DeepPro remains an explicit delayed adapter: no torch/cv2 import, download, or weight discovery occurs automatically. This round did not run a scoring container or submit to Codabench. See `docs/competition_research.md`, `docs/hardware_estimate.md`, and `SERVER_AGENT_REPORT.md`.

## YOLOv8-seg GPU workflow

The released data layout is `sequence/img` plus `sequence/mask`. The YOLO path keeps
the original images in place and creates only symlinks and labels:

```bash
python -m jinsight_track1.cli yolo-prepare /data2/2025/ldh/SatVideoIRSDT_v1_train_val/train \
  work_dirs/yolov8n_seg/yolo_dataset \
  --val-root /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val
python -m jinsight_track1.cli yolo-train \
  /data2/2025/ldh/SatVideoIRSDT_v1_train_val/train work_dirs/yolov8n_seg \
  --val-root /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val \
  --weights work_dirs/yolov8n_seg/yolov8n-seg.pt --download-weights
python -m jinsight_track1.cli yolo-eval work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
  /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val work_dirs/yolov8n_seg
```

When the official test directory is released, pass it explicitly to `yolo-infer`:

```bash
python -m jinsight_track1.cli yolo-infer MODEL.pt /path/to/test work_dirs/yolov8n_seg/test_predictions
```

The generated ZIP is validated for direct-root TXT entries, five-digit frame
numbers, counts, and coordinate order (`--coordinate-order xy|yx`, default `xy`).
Validation metrics are local point-matching proxies and are not official scores.

On this managed host, launch GPU commands through `scripts/gpu_session.sh` so
the NVIDIA character devices are recreated in the same session:

```bash
bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-infer MODEL.pt /path/to/test work_dirs/yolov8n_seg/test_predictions
```
