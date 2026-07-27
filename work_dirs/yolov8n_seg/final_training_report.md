# CSIG Track 1 YOLOv8-seg Recovery Report

## Status
- Training: complete, 50 epochs, existing `best.pt` reused.
- Local proxy evaluation: complete (`local proxy; not official scorer`).
- Official test submission: not complete; no official test directory was provided and no Codabench upload was performed.

## Data And Weights
- Train: `/data2/2025/ldh/SatVideoIRSDT_v1_train_val/train` (1433 sequences, 122887 frames)
- Val: `/data2/2025/ldh/SatVideoIRSDT_v1_train_val/val` (255 sequences, 23087 frames)
- Best weight: `work_dirs/yolov8n_seg/ultralytics/weights/best.pt`
- Best weight SHA256: `f923c668478f1d478034d6f32bc8b2c54fafdddb3a5daa11c98facb7ca69ee19`
- Last weight SHA256: `1b4253f7693fd56055cbfe392f87b00e8a5d9876adb0b5caf3b5749d15494712`
- Git commit: `e292680`

## Training Log
- Epochs completed: 50
- Elapsed seconds in `results.csv`: 41762.8
- NaN/Inf in `results.csv`: no
- Last epoch train losses: box 2.21779, seg 2.99441, cls 1.54988, dfl 0.63140
- Ultralytics mask metrics: precision 0.12946, recall 0.17187, mAP50 0.07124, mAP50-95 0.02713

## Local Proxy Evaluation
- Output: `work_dirs/yolov8n_seg/score_recovery_full_b4/val_proxy_metrics.json`
- Best confidence: 0.01
- Radius: 2.0 px
- Precision: 0.466002
- Recall: 0.293349
- F1: 0.360047
- TP/FP/FN: 25810 / 29576 / 62174
- Note: `batch=16` failed with CUDA OOM on full validation; `batch=4` completed.

## Validation ZIP
- Output dir: `work_dirs/yolov8n_seg/val_predictions_coordfix_v2_b4`
- ZIP: `work_dirs/yolov8n_seg/val_predictions_coordfix_v2_b4.zip`
- ZIP SHA256: `27c216012c31002af22382ca84fe828e2c188b4a285d733ba46e83f864945ad9`
- TXT count / ZIP entries: 255 / 255
- Frames checked: 23087
- Coordinates checked: 55386
- Coordinate order: `xy`
- Centroid coordinates: original-image pixels
- Format checks: passed
- Track IDs: `0`

## Submission Notes
- The generated validation ZIP is for recovery validation, not an official test submission.
- Test labels were not read, inferred, or generated.
- When the official test directory is provided, reuse the same `best.pt`, `imgsz=640`, `conf=0.01`, and `coordinate-order=xy`, then revalidate ZIP structure before upload.
