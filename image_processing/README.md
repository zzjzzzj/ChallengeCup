# 图像处理模块

本目录负责训练前的图像与数据处理，不负责模型训练。

主要功能：

- 审计图像、YOLO标签、重复项和数据质量；
- 生成稳定的 train/val/test 划分；
- 提取灰度、纹理、LBP、GLCM和频域特征；
- 根据真实框生成目标裁剪；
- 生成目标检测清单；
- 生成原始/增广对比使用的统一79/76评测协议。

查看命令：

```powershell
python -m image_processing.cli --help
```

典型流程：

```powershell
python -m image_processing.cli audit `
  --dataset "D:\datasets\datasets_r1_base_train" `
  --output scene_module/artifacts

python -m image_processing.cli features `
  --index scene_module/artifacts/scene_index.csv `
  --output scene_module/artifacts/scene_features.csv

python -m image_processing.cli comparison `
  --dataset-root "D:\datasets\yolo_augmented" `
  --output detector_module/artifacts/comparison_dataset
```

所有生成数据均写入已被 `.gitignore` 排除的 `artifacts/` 目录，只保留在本机。
