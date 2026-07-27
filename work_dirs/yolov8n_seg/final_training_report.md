# CSIG Track 1 YOLOv8-seg Training Report

## Data & Configuration
- **Train Sequences**: 1433
- **Val Sequences**: 255
- **Data Path**: /data2/2025/ldh/SatVideoIRSDT_v1_train_val
- **Config**: train_config.json (50 epochs, imgsz 640)

## Training Status
- **Epochs Completed**: 50
- **NaN/Inf**: No
- **Weights Path**: work_dirs/yolov8n_seg/ultralytics/weights/best.pt

## Metrics
- **Ultralytics Val**: Mask mAP50 ~ 0.8207 (from results.csv epoch 50)
- **Local Proxy Metrics**: The yolo-eval script timed out/stuck. Proxy metrics are unavailable at this moment.

## Validation Predictions
- **TXT/ZIP Verification**: Passed. 255 TXT files directly at the root of val_predictions.zip. No subdirectories found. Checked for 5-digit frame IDs.
- **Coordinates**: xy coordinate order was used, but the old export calculated centroids on the inference mask canvas and did not verify restoration to original-image pixels. Old prediction ZIP files must not be reused.

## Notes
- **Online incident update (2026-07-27)**: A later detection submission produced Final Score 0.81 and F1 0.0081; Track Accuracy and Track Completeness were both 0.
- **Root cause**: The legacy export used `result.masks.data` coordinates directly. The coordinate-recovery code and rerun plan are documented in `docs/score_recovery_and_next_training_plan.md`.
- **Submission status**: The old ZIP is invalid. Re-evaluate the existing `best.pt`, select confidence by point F1, and generate a new coordinate-fixed ZIP before any retraining.
