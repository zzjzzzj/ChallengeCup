# 目标检测基线模块

本模块对标赛题“基础目标检测识别性能”指标，并为后续类别增量学习和昇腾 310B 部署提供统一入口。

## 当前设计

- 检测器：预训练 YOLOv8n，4 类输出为 `soldier`、`small_aircraft`、`warship`、`tank`。
- 输入尺寸：640×640，保留原始 640×512 图像中的小目标信息。
- 数据划分：复用场景模块按“传感器×场景×文件序列”隔离的划分，不重新随机拆分相邻帧。
- 图像增强：默认关闭色相、饱和度增强；保留亮度扰动、轻微旋转、平移、尺度变化、水平翻转和 Mosaic。
  做离线增广对照实验时用 `--no-builtin-aug` 把这些在线增广全部归零，见下文。
- 指标：同时保存 mAP@0.5、mAP@0.5:0.95、Precision、Recall、分类别 AP 和推理耗时。

## 数据情况

运行 `prepare_detection_dataset.py` 后会进行完整图片与 YOLO 标签校验。当前实际读取结果为：

| 划分 | 图片 | 标注框 |
|---|---:|---:|
| train | 525 | 2025 |
| val | 114 | 459 |
| test | 111 | 473 |
| 合计 | 750 | 2957 |

数据说明文档中的标注总数为 2956，而当前文件实测为 2957；后续文档和实验统一以实际标签扫描结果为准，并保留这一差异说明。

## 安装依赖

先按本机 CUDA 版本安装 PyTorch 和 torchvision，再运行：

```powershell
python -m pip install -r .\scene_recognition\detector_module\requirements.txt
```

## 一键运行

```powershell
powershell -ExecutionPolicy Bypass -File .\scene_recognition\detector_module\run_baseline.ps1
```

可调整训练轮数和批大小：

```powershell
.\scene_recognition\detector_module\run_baseline.ps1 -Epochs 100 -BatchSize 16
```

## 分步运行

生成数据配置并验证全部标签：

```powershell
python -m scene_recognition.detector_module.prepare_detection_dataset
```

训练并在独立测试集评估：

```powershell
python -m scene_recognition.detector_module.train_detector `
  --epochs 100 `
  --batch-size 16 `
  --name yolov8n_baseline_v1
```

按 IR/SAR 和四类场景分别评估：

```powershell
python -m scene_recognition.detector_module.evaluate_detector `
  --model .\scene_recognition\detector_module\runs\yolov8n_baseline_v1\weights\submission_map50.pt
```

训练框架默认按照 mAP@0.5:0.95 保存 `best.pt`。如正式评分采用 mAP@0.5，需先在已保存权重中按照验证集 mAP@0.5 选择提交权重：

```powershell
python -m scene_recognition.detector_module.select_baseline_checkpoint `
  --run .\scene_recognition\detector_module\runs\yolov8n_baseline_v1
```

### 离线增广数据集对照实验

团队产出的本地增广数据集包含训练 4400 张（595 张原图 + 3805 张派生图），
验证 155 张全部为未增广原图）。要回答“这套增广到底有没有用”，必须让**训练集成为唯一变量**：

- 两组都加 `--no-builtin-aug`，把 YOLO 自带的 mosaic/fliplr/degrees/scale/hsv 等
  15 项在线增广全部归零。否则测到的是“离线增广 + 在线增广”的混合效果，无法归因。
- 两组共用同一份 155 张验证集。
  **不要拿仓库里旧的 `detection_dataset`（525/114/111 划分）当对照组**——
  那套划分与增广数据集的验证集只有 6 张重合，指标不可比。
- 增广组 4400 张、对照组 595 张，同样 epoch 下增广组的梯度步数是 7.4 倍。
  因此除了等 epoch 组，还要跑一个等迭代对照组（595×444 ≈ 4400×60），
  用来区分“增广有用”和“只是训练得更久”。

```powershell
# 对照组：595 张原图，60 轮
python -m scene_recognition.detector_module.train_detector --data scene_recognition/detector_module/configs/data_noaug.yaml `
  --epochs 60 --patience 999 --batch-size 16 --workers 2 --no-builtin-aug --name ab_noaug_e60

# 增广组：4400 张，60 轮（等 epoch）
python -m scene_recognition.detector_module.train_detector --data scene_recognition/detector_module/configs/data_augmented.yaml `
  --epochs 60 --patience 999 --batch-size 16 --workers 2 --no-builtin-aug --name ab_augmented_e60

# 对照组：595 张原图，444 轮（等迭代，595×444 ≈ 4400×60）
python -m scene_recognition.detector_module.train_detector --data scene_recognition/detector_module/configs/data_noaug.yaml `
  --epochs 444 --patience 999 --batch-size 16 --workers 2 --no-builtin-aug --name ab_noaug_e444
```

`--patience 999` 是为了让三组都跑满各自的预算，不被早停在不同轮次截断。

汇总成对照表（会自动标注预算是否对齐、增广开关是否一致）：

```powershell
python -m scene_recognition.detector_module.compare_augmentation `
  --run 增广60轮=scene_recognition/detector_module/runs/ab_augmented_e60 `
  --run 对照60轮=scene_recognition/detector_module/runs/ab_noaug_e60 `
  --run 对照444轮=scene_recognition/detector_module/runs/ab_noaug_e444 `
  --train-images 增广60轮=4400 --train-images 对照60轮=595 --train-images 对照444轮=595 `
  --output docs/增广对照实验.md
```

增广数据集的配置只有 train/val 没有 test，脚本会自动在 val 上评估，
并把口径写进 `baseline_summary.json` 的 `evaluation_split` 字段，不会静默冒充测试集指标。

### 从零训练对照

`--no-pretrained` 用于随机初始化对照。注意它**不接受权重文件**：
`YOLO('yolov8n.pt')` 在构造时就已把 COCO 权重灌进网络，
靠 `train(pretrained=False)` 清零只是当前 ultralytics 版本的实现细节（`--resume` 时完全失效）。
因此脚本强制从零训练走 `.yaml` 架构，显式传 `.pt` 又要求从零时直接报错。

生成增量学习协议：

```powershell
python -m scene_recognition.detector_module.create_incremental_protocol
```

导出 ONNX 并在独立测试集验证转换精度：

```powershell
python -m scene_recognition.detector_module.export_detector `
  --model .\scene_recognition\detector_module\runs\yolov8n_baseline_v1\weights\submission_map50.pt
```

## 关键产物

- `artifacts/detection_dataset/dataset.yaml`：YOLO 数据配置。
- `artifacts/detection_dataset/dataset_stats.json`：实测数据与标签统计。
- `configs/incremental_protocol.json`：基础阶段和两轮模拟增量阶段定义。
- `runs/<实验名>/weights/best.pt`：验证集最优权重。
- `runs/<实验名>/weights/submission_map50.pt`：仅根据验证集 mAP@0.5 选出的评分权重。
- `runs/<实验名>/checkpoint_selection.json`：权重选择依据与对应测试指标。
- `runs/<实验名>/baseline_summary.json`：独立测试集指标和环境信息。
- `runs/<实验名>/slice_evaluation/`：分传感器、分场景指标。
- `runs/<实验名>/exports/`：ONNX 模型、转换验证指标和 ATC 命令模板。

## 实验纪律

1. 训练只使用 train，模型选择只使用 val，最终数字只在 test 上报告。
2. 每次实验必须使用新的运行名称，不覆盖已有有效结果。
3. 正式增量实验前，需要向主办方确认是否允许保存旧类图片、特征或回放缓存。
4. 当前 air 场景没有 SAR 数据，不能声称模型已验证 SAR-air 泛化能力。
5. 主办方尚未明确 mAP 口径时，同时报告 mAP@0.5 和 mAP@0.5:0.95。
## Additional tools

Diagnose detector weak spots by class, sensor, scene, object size, and top missed samples:

```powershell
python -m scene_recognition.detector_module.diagnose_detector `
  --model .\scene_recognition\detector_module\runs\yolov8n_baseline_v1\weights\submission_map50.pt
```

Outputs:

- `runs/<run_name>/error_diagnosis/diagnosis.json`
- `runs/<run_name>/error_diagnosis/class_summary.csv`
- `runs/<run_name>/error_diagnosis/slice_summary.csv`
- `runs/<run_name>/error_diagnosis/size_summary.csv`
- `runs/<run_name>/error_diagnosis/top_missed_samples.csv`

Build materialized YOLO views for the simulated incremental-learning protocol:

```powershell
python -m scene_recognition.detector_module.build_incremental_dataset
```

Outputs:

- `artifacts/incremental_dataset/incremental_dataset_summary.json`
- `artifacts/incremental_dataset/stage_*/train_new.yaml`
- `artifacts/incremental_dataset/stage_*/train_replay.yaml`
