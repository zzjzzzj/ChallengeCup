# ChallengeCup 多模态遥感场景分类与目标检测

本仓库实现了红外与 SAR 遥感图像上的场景分类、目标分类和端到端目标检测，并记录了从真实框裁剪分类到 ResNet18/YOLOv8n 同口径检测的完整实验过程。

## 项目任务

- 场景分类：识别 `air`、`sea`、`urban`、`forest`。
- 目标分类：在已知真实框时识别 `soldier`、`small_aircraft`、`warship`、`tank`。
- 目标检测：从整图输出目标框、类别和置信度。
- 对比实验：原始/增广训练集 × 预训练/从零训练 × ResNet18/YOLOv8n。

## 主要结果

| 任务 | 评测范围 | 主要结果 | 说明 |
|---|---|---|---|
| 场景四分类 | 111 张独立测试图 | Accuracy **92.79%**，Macro-F1 **92.25%** | ANOVA 30 特征 + RBF-SVM |
| ResNet18 真实框裁剪分类 | 473 个测试目标 | Accuracy **99.79%** | 已知位置，不包含定位能力 |
| ResNet18-FPN 端到端检测 | 76 张独立测试图 | 最佳 mAP@0.5 **87.20%** | 增广训练集 + ImageNet 预训练 |
| YOLOv8n 端到端检测 | 同一批 76 张测试图 | 最佳 mAP@0.5 **85.25%** | 与 ResNet18 使用同一 AP 评测器复算 |

最终检测主表使用同一批 76 张测试图和同一套 COCO-style 101-point AP 实现。裁剪分类 Accuracy、整图存在判断 Exact Match 和检测 mAP 衡量的任务不同，不能直接比较大小。

## 代码结构

```text
ChallengeCup/
├─ train.py                              # 统一训练与数据准备入口
├─ run_detection_experiments.py          # 最终八组端到端检测调度器
├─ scene_module/                         # 场景数据审计、特征提取、训练与推理
├─ target_classifier_module/             # ResNet18裁剪分类与历史整图存在判断
├─ detector_module/                      # YOLOv8n和ResNet18-FPN检测
│  ├─ prepare_comparison_dataset.py      # 生成最终原始/增广对比协议
│  ├─ train_detector_ablation.py         # 单组YOLOv8n训练
│  ├─ resnet18_detector.py               # 单组ResNet18-FPN训练
│  └─ evaluate_yolo_same_protocol.py     # YOLO同评测器复算
├─ docs/                                 # 实验记录、诊断和结果报告
└─ requirements.txt                      # Python依赖，PyTorch除外
```

根目录旧脚本 `run_comparison_experiments.py` 保留早期 155 张验证集协议，用于复现历史实验；新的正式检测实验请使用 `run_detection_experiments.py` 或 `python train.py detection-matrix`。

## 环境安装

推荐 Python 3.11 及以上版本。先根据本机 CPU/CUDA 环境从 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装匹配的 `torch` 和 `torchvision`，再安装仓库依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 按 PyTorch 官网给出的命令安装 torch 和 torchvision
pip install -r requirements.txt
```

查看统一入口：

```powershell
python train.py --help
python train.py detection-matrix --help
```

## 数据目录

原始数据和训练权重不包含在仓库中。场景基础数据应包含图像和同名 YOLO 标签；最终增广对比数据使用以下结构：

```text
yolo_augmented/
├─ images/
│  ├─ train/          # 原图和文件名含 __aug- 的增广图
│  └─ val/            # 155张未增广留出图
└─ labels/
   ├─ train/
   └─ val/
```

每张图片必须有同名 `.txt` YOLO 标签。数据生成器会：

- 将训练目录中不含 `__aug-` 的文件作为原始训练组；
- 将全部训练文件作为增广训练组；
- 按场景和文件顺序把未增广留出图交替拆成 val/test；
- 生成本机绝对路径清单和两份数据 YAML，避免提交不可移植的路径。

## 复现场景分类

### 1. 数据审计与划分

```powershell
python train.py scene-prepare `
  --dataset "D:\datasets\datasets_r1_base_train" `
  --output scene_module/artifacts
```

### 2. 特征提取

```powershell
python train.py scene-extract `
  --index scene_module/artifacts/scene_index.csv `
  --output scene_module/artifacts/scene_features.csv
```

### 3. 特征消融与最终模型训练

```powershell
python train.py scene-evaluate `
  --features scene_module/artifacts/scene_features.csv `
  --output scene_module/runs/feature_eval_report
```

输出包括 `feature_ablation.csv`、`model_metadata.json`、混淆矩阵、预测明细和 `scene_feature_svm.joblib`。

## 复现最终八组目标检测

### 1. 生成统一对比协议

```powershell
python train.py prepare-comparison `
  --dataset-root "D:\datasets\yolo_augmented" `
  --output detector_module/artifacts/comparison_dataset
```

预期生成：

```text
detector_module/artifacts/comparison_dataset/
├─ train_noaug.txt
├─ train_aug.txt
├─ val.txt
├─ test.txt
├─ data_noaug.yaml
├─ data_aug.yaml
└─ dataset_stats.json
```

先检查将要运行的八组命令：

```powershell
python train.py detection-matrix --dry-run
```

确认路径和显存配置后启动训练：

```powershell
python train.py detection-matrix
```

八组实验为：

| 模型 | 训练数据 | 初始化 |
|---|---|---|
| YOLOv8n | 原始 / 增广 | COCO预训练 / 从零训练 |
| Faster R-CNN + ResNet18-FPN | 原始 / 增广 | ImageNet预训练 / 从零训练 |

已完成的组默认按 `metrics.json` 或 `ablation_summary.json` 自动跳过。可使用 `--only` 运行指定组，使用 `--force` 强制重跑。

```powershell
python train.py detection-matrix `
  --only cmp8_yolov8n_aug_pretrained

python train.py detection-matrix `
  --model resnet
```

### 2. 单独训练一个模型

YOLOv8n：

```powershell
python train.py yolo `
  --data detector_module/artifacts/comparison_dataset/data_aug.yaml `
  --name custom_yolov8n_aug_pretrained `
  --epochs 150 `
  --eval-split test `
  --exist-ok
```

从零训练时增加 `--no-pretrained`。

ResNet18-FPN：

```powershell
python train.py resnet-detector `
  --data detector_module/artifacts/comparison_dataset/data_aug.yaml `
  --output detector_module/runs/custom_resnet18det_aug_pretrained `
  --epochs 6 `
  --batch-size 4
```

从零训练时增加 `--no-pretrained`。

### 3. 统一评测器复算 YOLO

四个标准命名的 YOLO 运行完成后执行：

```powershell
python train.py yolo-evaluate
```

该命令使用与 ResNet18 检测器相同的 mAP 实现复算 YOLO，生成最终横向比较所需指标。

## 辅助实验：真实框裁剪分类

该实验用于测量“已知正确位置后能否分对类别”，不用于替代检测 mAP。

```powershell
python train.py crop-prepare `
  --index scene_module/artifacts/scene_index.csv `
  --output target_classifier_module/artifacts/target_crops

python train.py crop-classifier `
  --manifest target_classifier_module/artifacts/target_crops/manifest.csv `
  --output target_classifier_module/runs/resnet18_target_baseline_none `
  --epochs 12 `
  --batch-size 32
```

历史整图存在判断仍可通过 `python train.py whole-classifier --help` 运行，但该任务已确认存在场景捷径，不作为目标检测结论。

## 输出文件

| 文件 | 内容 |
|---|---|
| `best.pt` | 验证集选择的最佳权重 |
| `last.pt` | 最后一轮权重，部分训练器提供 |
| `history.csv` / `results.csv` | 逐轮训练记录 |
| `metrics.json` | ResNet18分类或检测指标 |
| `ablation_summary.json` | YOLO训练配置和最终指标 |
| `matrix_log.json` | 八组训练状态、耗时和日志路径 |

训练输出默认位于模块的 `runs/` 目录，该目录不会提交 Git。

## 实验文档

- [实验记录精简版](docs/完整实验记录-精简版.md)
- [按研究过程整理的完整实验记录](docs/完整实验记录-按研究过程.md)
- [全部运行目录与测试集口径](docs/实验组数与准确率汇总.md)
- [数据集分类与统计](docs/数据集分类与统计.md)
- [本地运行与 Docker 部署](docs/运行与部署.md)
- [脱敏训练结果](docs/results/README.md)
- [端到端检测答辩版报告](docs/comparison/端到端检测对比报告-答辩版.md)
- [场景捷径与模型选择诊断](docs/诊断报告-场景捷径与模型选择缺陷.md)

## 测试

```powershell
python -m unittest discover -v
```

测试覆盖数据划分、YOLO标签转换、ResNet18训练与评价、检测指标、增量协议和场景捷径诊断。真实训练需要本地数据和可用的 PyTorch 运行环境。

## 上传 GitHub 前检查

- 不上传赛事原始图像、标签、增广数据或任何含绝对本机路径的数据清单。
- 不上传 `.pt`、`.onnx`、`.om`、`.joblib` 等模型文件。
- 不上传 `runs/`、训练日志、临时目录和本地缓存。
- 赛事 PDF/Word 文档只保留在本机，已从发布历史和Git跟踪范围排除。
- 训练结果只提交 `docs/results/` 中的脱敏 JSON 或整理后的 Markdown，不提交预测明细、原始 CSV 和大体积权重。
