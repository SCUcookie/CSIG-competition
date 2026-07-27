# 榜单分数恢复与下一轮训练计划

更新日期：2026-07-27

## 1. 已确认的线上结果

- Codabench Final Score：`0.81`
- Codabench F1 Score：`0.0081`
- Track Accuracy：`0`
- Track Completeness：`0`
- 本地 Ultralytics 指标：Mask mAP50 `0.8207`

结论：线上 `0.0081` 是质心点匹配 F1，本地 `0.8207` 是分割
Mask mAP50，两者不是同一指标。Track 两项为 0 是当前检测模式使用
`track_id=0` 的预期结果，现阶段不应优先处理轨迹奖励。

## 2. 首要故障与代码修复

旧版 `_prediction_points` 直接在 `result.masks.data` 上计算质心。
该张量可能位于 `imgsz`/letterbox 推理画布，而提交文件要求原图像素坐标。
在 256、512×640、约 742×733、1024 等混合分辨率上直接写出这些坐标，
会使绝大多数预测点无法与真值匹配。

本次代码修改：

1. 使用 Ultralytics `result.masks.xy` 的原图像素多边形计算面积质心。
2. 极小或退化多边形回退到原图坐标下的 `result.boxes.xywh` 中心。
3. 写出前检查有限值和 `0 <= x < width, 0 <= y < height`。
4. `yolo-eval` 和 `yolo-infer` 改为流式批量推理，避免逐帧调用。
5. `yolo-eval --conf-grid` 一次推理扫描多个置信度，并按原图分辨率报告 F1。
6. 推理和验证均显式支持 `--imgsz`，确保训练与推理设置一致。

官方说明采用“横坐标、纵坐标”，恢复提交默认使用 `--coordinate-order xy`。
在没有新的官方证据前不要改成 `yx`。

## 3. 服务器恢复执行顺序

本阶段禁止启动新训练。先复用现有 `best.pt` 恢复正确的提交链路。

### 3.1 拉取与基础检查

```bash
cd /path/to/CSIG-repository
git pull --ff-only
python -m pip install -e .
pytest -q
```

所有 GPU 命令均通过托管会话执行：

```bash
bash scripts/gpu_session.sh nvidia-smi
```

### 3.2 小规模坐标回归

先跑少量序列，确认代码、权重和显存正常：

```bash
bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-eval \
  work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
  /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val \
  work_dirs/yolov8n_seg/score_recovery_smoke \
  --device 0 --imgsz 640 --batch 16 --radius 2 \
  --conf-grid 0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.70 \
  --max-sequences 4 --max-frames 20
```

必须检查：

- 命令正常结束，不出现越界质心异常。
- `val_proxy_metrics.json` 中 `centroid_coordinates` 为
  `original-image pixels`。
- 四种原始分辨率均由后续全量评估覆盖。
- 抽样将预测点画回原图时，点位于目标附近，而不是按 640 画布缩放。

### 3.3 全量验证和阈值选择

```bash
bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-eval \
  work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
  /data2/2025/ldh/SatVideoIRSDT_v1_train_val/val \
  work_dirs/yolov8n_seg/score_recovery_full \
  --device 0 --imgsz 640 --batch 16 --radius 2 \
  --conf-grid 0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.70
```

读取：

```text
work_dirs/yolov8n_seg/score_recovery_full/val_proxy_metrics.json
```

记录 `best_confidence`、Precision、Recall、F1、TP、FP、FN，以及
`best_resolution_breakdown`。本地半径 2 像素只是代理口径，不能宣称为官方分数。

### 3.4 重新生成提交

将下面的 `<BEST_CONF>` 替换为全量验证得到的 `best_confidence`。必须使用全新
输出目录，避免旧 TXT 混入 ZIP。

```bash
bash scripts/gpu_session.sh python -m jinsight_track1.cli yolo-infer \
  work_dirs/yolov8n_seg/ultralytics/weights/best.pt \
  /path/to/official_test \
  work_dirs/yolov8n_seg/test_predictions_coordfix_v2 \
  --device 0 --imgsz 640 --batch 16 --conf <BEST_CONF> \
  --coordinate-order xy
```

待提交文件：

```text
work_dirs/yolov8n_seg/test_predictions_coordfix_v2.zip
```

提交前确认：

- ZIP 根目录直接包含所有 TXT，无子目录。
- TXT 数量与测试序列数一致。
- 每帧均有一行，空帧写为 `帧号 0`。
- 当前检测提交所有目标 ID 均为 0。
- 随机检查四类分辨率的坐标均在原图边界内。
- 保存 ZIP SHA256、所用权重 SHA256、Git commit、`imgsz` 和置信度。

## 4. 恢复阶段的判定

1. 如果本地点 F1 从接近 0 恢复到正常量级，先提交坐标修复版本，不训练。
2. 如果本地 F1 正常但线上仍接近 0，依次检查测试序列命名、帧编号、ZIP
   层级和官方坐标样例；不要首先怀疑模型精度。
3. 如果线上 F1 与本地趋势一致但仍不够高，再进入下一轮训练。
4. Track 两项继续为 0 不阻塞检测主分恢复。只有检测 F1 稳定后才启用跟踪 ID。

## 5. 下一轮训练实验

所有实验使用相同的序列级 train/val 划分。模型选择、置信度选择和早停判断
以验证集质心点 F1 为主，Mask mAP50 只作辅助诊断。

| 实验 | 模型/设置 | 目的 |
| --- | --- | --- |
| E0 | 现有 YOLOv8n-seg，640 | 坐标修复后的强制基线，不训练 |
| E1 | YOLOv8n-seg，1024 | 检查高分辨率对微小目标定位的收益 |
| E2 | YOLOv8s-seg，1024 | 检查容量提升，和 E1 保持其他条件一致 |
| E3 | 最优单帧模型 + 时序滤波 | 降低孤立 FP、补短暂漏检 |
| E4 | 稳定检测后启用轨迹 ID | 最后争取 Track Accuracy/Completeness 奖励 |

训练要求：

- 每个实验使用独立 `work_dir`，保留 config、results.csv、best.pt 和权重哈希。
- 先用少量序列验证 1024 输入显存与吞吐，再开始全量训练。
- 对 640 与 1024 使用相同验证序列和同一套置信度扫描。
- 分别报告 256、512×640、约 742×733、1024 分辨率的点 F1。
- 若高分辨率仅提高 Mask mAP、没有提高点 F1，不进入下一实验。
- 不使用线上测试结果反向选择大量超参数，保留最终阶段提交额度。

## 6. 当前优先级

```text
原图坐标修复
  -> 现有权重全量点 F1 与置信度扫描
  -> 新 ZIP 在线复测
  -> 1024 输入实验
  -> 更大模型
  -> 时序与跟踪奖励
```

