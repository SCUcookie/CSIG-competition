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
- **Coordinates**: xy coordinate order was used. Checked and verified non-empty frames.

## Notes
- **Testing**: Test set is not open yet. No test labels have been read or used.
- **Submission**: Nothing has been uploaded to Codabench.
