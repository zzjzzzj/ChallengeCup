# 训练数据与配置细节

八组对比实验的数据构成、超参配置、运行环境与耗时。
识别准确率结果见 `accuracy_by_scene_and_class.md`。

## 一、实验矩阵

三个自变量各两档，2×2×2 = 8 组：

| 自变量 | 档位 A | 档位 B |
|---|---|---|
| 模型 | ResNet18（真值框裁剪四类分类） | YOLOv8n（端到端检测） |
| 训练数据 | 原始集 | 离线增广集 |
| 权重初始化 | ImageNet / COCO 预训练 | 随机初始化（从零） |

**唯一变量控制**：增广组与原始组**共用同一批验证数据**；原始组就是增广集中
文件名不含 `__aug-` 的那部分原图。两组之间唯一差异即「有没有离线增广」。

> 不要拿仓库中旧的 `detection_dataset`（525/114/111 划分）作对照 ——
> 它与本次的 155 张验证集只有 6 张重合，指标不可比。

---

## 二、数据构成

### 2.1 YOLOv8n 检测数据（整图）

| 划分 | 原始集 | 增广集 |
|---|---|---|
| 训练图像 | **595** | **4400** |
| ├ IR | 429 | 2574 |
| └ SAR | 166 | 1826 |
| 验证图像 | **155**（共用，全部未增广） | **155**（同左） |

验证集场景分布：forest 43、urban 42、air 35、sea 35。

### 2.2 ResNet18 分类数据（按真值框裁剪，padding 10%）

| 划分 | 原始集 | 增广集 |
|---|---|---|
| 训练裁剪 | **2388** | **17663** |
| ├ small_aircraft | 558 | 3348 |
| ├ soldier | 492 | 3812 |
| ├ tank | 758 | 5848 |
| └ warship | 580 | 4655 |
| 验证裁剪 | **569**（共用） | **569**（同左） |

原始集训练裁剪的其他切分：IR 1721 / SAR 667；air 558、forest 591、sea 580、urban 659。

验证裁剪（569）构成：

| 场景 | small_aircraft | soldier | tank | warship | 小计 |
|---|---|---|---|---|---|
| air | 141 | 0 | 0 | 0 | 141 |
| forest | 0 | 60 | 91 | 0 | 151 |
| sea | 0 | 0 | 0 | 122 | 122 |
| urban | 0 | 67 | 88 | 0 | 155 |
| **合计** | 141 | 127 | 179 | 122 | **569** |

模态：IR 421 / SAR 148。

### 2.3 数据来源

增广数据集位于团队本地数据根目录，由离线增广脚本生成（见该目录下
`图像增强处理流程说明.txt` 与 `augmentation_manifest.csv`）。
裁剪由 `scene_recognition/target_classifier_module/prepare_crops_from_yolo_dir.py` 生成，
padding_ratio = 0.1，缺失标签数 0。

---

## 三、超参配置

### 3.1 ResNet18

| 参数 | 取值 |
|---|---|
| 训练轮数 | 40（固定，不早停） |
| batch size | 32 |
| 输入尺寸 | 224 × 224 |
| 学习率 | 3e-4 |
| weight decay | 1e-4 |
| 优化器 | Adam |
| 随机种子 | 42 |
| num_workers | 0 |
| 在线增广 | **关闭**（`--augmentation none`） |
| 骨干冻结轮数 | 0 |
| 模型选择 | 验证集 Macro-F1 最高；同分取验证 loss 最低 |

预训练组加载 `torchvision` ImageNet 权重；从零组 `--no-pretrained` 随机初始化。

### 3.2 YOLOv8n

| 参数 | 取值 |
|---|---|
| 训练轮数上限 | 150 |
| patience（早停） | 20 |
| batch size | 16 |
| 输入尺寸 | 640 × 640 |
| 优化器 | auto |
| 学习率调度 | cosine (`cos_lr=True`) |
| 随机种子 | 42 |
| deterministic | True |
| AMP 混合精度 | 开启 |
| workers | 4 |
| 模型选择 | fitness（= mAP@0.5:0.95）最高 |

预训练组加载 `yolov8n.pt`（COCO）；从零组用 `yolov8n.yaml` 建结构 **且** `pretrained=False`
（二者缺一都会悄悄加载权重）。

**在线增广参数**（四组 YOLO 完全一致）：

| 参数 | 取值 | 参数 | 取值 |
|---|---|---|---|
| mosaic | 0.6 | fliplr | 0.5 |
| close_mosaic | 10 | flipud | 0.0 |
| degrees | 5.0 | hsv_h | 0.0 |
| translate | 0.08 | hsv_s | 0.0 |
| scale | 0.20 | hsv_v | 0.15 |
| mixup | 0.0 | | |

> 因为在线增广在增广组与原始组中**均开启且参数相同**，本实验量出的是
> **离线增广在在线增广之上的增量**，会小于离线增广的全部价值。

---

## 四、运行环境

| 项目 | 版本 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop |
| PyTorch | 2.9.1+cu128 |
| ultralytics | 8.4.100 |
| Python | 3.13 |
| 操作系统 | Windows 11 |

---

## 五、实际训练轮数与耗时

### 5.1 ResNet18（固定 40 轮）

| 数据 | 权重 | 训练轮数 | 选中轮次 | 耗时 |
|---|---|---|---|---|
| 原始集 | 预训练 | 40 | 第 8 轮 | 5.1 min |
| 原始集 | 从零 | 40 | 第 27 轮 | 5.3 min |
| 增广集 | 预训练 | 40 | 第 7 轮 | 31.2 min |
| 增广集 | 从零 | 40 | 第 18 轮 | 32.1 min |

单轮耗时：原始集 7.3 秒，增广集 46.3 秒。

### 5.2 YOLOv8n（patience 20 早停）

| 数据 | 权重 | 实际轮数 | 最佳轮次 | 单轮耗时 | 总耗时 |
|---|---|---|---|---|---|
| 原始集 | 预训练 | 46 | 第 26 轮 | 7.0 s | 7.1 min |
| 原始集 | 从零 | 133 | 第 113 轮 | 7.2 s | 17.5 min |
| 增广集 | 预训练 | 51 | 第 31 轮 | 59.5 s | 52.8 min |
| 增广集 | 从零 | 59 | 第 39 轮 | 55.5 s | 56.6 min |

**八组合计 3.5 小时**，全部成功，0 失败。

预训练显著加快收敛：ResNet18 第 7~8 轮达到最佳（从零需 18~27 轮）；
YOLOv8n 第 26 轮达到最佳（从零需 113 轮，慢 4.3 倍）。

---

## 六、已知协议局限（汇报须声明）

1. **ResNet18 的 test 复用了 val。** 增广数据集只提供 train/val 两个划分，
   而训练脚本强制要求 test 划分，故 test 复用同一批 569 张裁剪。
   模型选择与最终汇报在同一批数据上进行，**绝对值偏乐观**。
   八组协议完全一致，横向差值仍可信。YOLOv8n 一侧统一在 val 上评测，不受此影响。

2. **单种子（seed=42）**，未做多种子重复，差值未附标准差。
   历史记录显示该任务种子噪声标准差约 0.61 个百分点，
   故 ResNet18 一侧 +0.70 以下的差值不宜强解读。

3. **数据集存在场景—类别强绑定。** air 场景只含 small_aircraft、sea 场景只含 warship，
   forest 与 urban 仅 soldier / tank 二选一。一个「场景 one-hot + 框宽高」的浅决策树
   不看任何像素即可达到 **97.67%** 准确率。ResNet18 的所有数字须减去这一平凡上限来解读。
   YOLOv8n 的 mAP 不受此影响。

4. **两个模型的指标不可横向比较。** ResNet18 的定位由真值框白送，
   YOLOv8n 必须自己找到目标再分类。要同口径比较需运行
   `scene_recognition/detector_module/gt_box_classification.py`。

---

## 七、复现命令

```bash
python run_comparison_experiments.py
python collect_comparison.py
```

已完成的组会被自动跳过，中断后重跑即可续上。

**注意**：`run_comparison_experiments.py` 中 ResNet18 的 `--num-workers` 必须保持为 `0`。
`training.py` 建了 train/val/test 三个 DataLoader 且未设 `persistent_workers`，
Windows 用 spawn 起 worker、每轮重建一遍，单个 worker 光 import torch 就约 9 秒。
实测 `num_workers=4` 时 2388 张裁剪跑一轮需 300 秒且 GPU 利用率 0%；
改为 `0` 后 17663 张只需 79 秒，等效快约 28 倍。

## 八、产物索引

| 内容 | 路径 |
|---|---|
| 分场景/分目标准确率表 | `docs/comparison/accuracy_by_scene_and_class.md` |
| 自动汇总表 | `docs/comparison/comparison_report.md` |
| 汇总 CSV | `docs/comparison/comparison_results.csv` |
| YOLO 分场景原始指标 | `docs/comparison/yolo_scene_metrics.json` |
| 运行日志 | `docs/comparison/run_log.json`、`docs/comparison/logs/*.log` |
| ResNet18 权重与曲线 | `scene_recognition/target_classifier_module/runs/cmp_resnet18_*/` |
| YOLOv8n 权重与曲线 | `scene_recognition/detector_module/runs/cmp_yolov8n_*/` |
