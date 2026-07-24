# 训练完成后 Agent 执行 Prompt

你现在接手的是 CSIG Track 1 YOLOv8-seg 首轮训练后的收尾任务。请在当前项目目录继续工作，不要重新训练，除非权重不存在或已损坏。

## 目标

完成训练结果核验、本地验证评测、验证集提交 ZIP 生成，并准备测试集开放后的可复用推理命令。当前验证结果只能标记为本地代理指标，不得宣称为官方成绩；不要上传 Codabench。

## 执行步骤

1. 通过 `bash scripts/gpu_session.sh ...` 运行所有 GPU 命令。确认 `nvidia-smi`、PyTorch CUDA 和 Ultralytics 可见 GPU。
2. 检查以下文件是否存在且可加载：
   - `work_dirs/yolov8n_seg/ultralytics/weights/best.pt`
   - `work_dirs/yolov8n_seg/ultralytics/weights/last.pt`
   - `work_dirs/yolov8n_seg/train_config.json`
   - `work_dirs/yolov8n_seg/ultralytics/results.csv`
3. 检查 `results.csv` 和训练日志：确认没有 NaN/Inf，记录最后 epoch 的 loss、box/seg 指标及 Ultralytics 验证指标。
4. 使用训练完成的 `best.pt` 对独立验证集运行本地代理评测：

   ```bash
   bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-eval \
     work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
     /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val \
     work_dirs/yolov8n_seg
   ```

   保存并报告 Precision、Recall、F1、TP、FP、FN，并明确写出 `local proxy; not official scorer`。

5. 生成完整验证集检测 TXT 和 ZIP，默认坐标顺序为 `xy`：

   ```bash
   bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-infer \
     work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
     /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val \
     work_dirs/yolov8n_seg/val_predictions \
     --device 0 --coordinate-order xy
   ```

6. 严格校验验证输出：
   - TXT 数量必须等于验证序列目录数量（当前预期 255）；
   - ZIP 必须有 255 个直接位于根目录的 TXT 条目；
   - 每行帧号为五位、目标数量与坐标字段匹配；
   - 不得产生子目录；
   - 随机抽查空帧、非空帧和坐标顺序；
   - 若发现坐标顺序不确定，保留 `xy` 和 `yx` 两个版本并在报告中说明，不能擅自宣称已确认。

7. 生成 `work_dirs/yolov8n_seg/final_training_report.json` 和一份简短 Markdown 报告，至少包括：
   - 训练配置、数据路径和 train/val 序列数；
   - 权重路径及 SHA256；
   - 训练完成 epoch、耗时和是否出现 NaN/Inf；
   - Ultralytics 验证指标；
   - 本地代理 Precision/Recall/F1；
   - TXT/ZIP 校验结果；
   - 测试集尚未开放、未使用测试标签、未上传 Codabench 的说明。

8. 测试集开放后，要求用户显式提供测试数据目录，再复用同一个 `best.pt` 运行：

   ```bash
   bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-infer \
     work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
     /path/to/official_test \
     work_dirs/yolov8n_seg/test_predictions \
     --device 0 --coordinate-order xy
   ```

   不读取、不推断、不生成任何测试标签；上传前必须再次确认官方样例要求的坐标顺序。

## 收尾要求

运行 `pytest -q` 和 CLI smoke。不要删除已有训练日志、权重、转换统计或验证输出。最终回复必须区分“训练已完成”“本地代理评测完成”和“官方测试提交未完成”三种状态。
