# 场景识别模块

本目录是仓库的主要训练模块，负责场景分类、目标切片分类、端到端目标检测、评估、推理和实验调度。

场景识别包含两条路线：

- 手工特征路线：StandardScaler + ANOVA特征筛选 + RBF-SVM；
- 图像模型路线：直接读取场景图像训练分类模型。

此外，本目录还包括：

- `target_classifier_module/`：直接使用真实目标框切片训练 ResNet18 分类器；
- `detector_module/`：训练和评估 YOLOv8、ResNet18-FPN 端到端检测器；
- `experiments/`：运行原始/增广数据、预训练/从零训练等实验矩阵。

图像审计、预处理、特征提取和训练清单生成位于 `../image_processing/`。数据集、模型权重和完整训练日志均保留在本地，不提交 Git。

查看命令：

```powershell
python -m scene_recognition.cli --help
```

手工特征模型训练：

```powershell
python -m scene_recognition.cli train-features `
  --features image_processing/artifacts/scene_features.csv `
  --output image_processing/runs/feature_eval_report
```

单图推理：

```powershell
python -m scene_recognition.cli infer `
  --image "$env:SAMPLE_IMAGE" `
  --model image_processing/runs/feature_eval_report/scene_feature_svm.joblib `
  --metadata image_processing/runs/feature_eval_report/model_metadata.json
```

启动本地分析台：

```powershell
python -m scene_recognition.cli dashboard
```

正式八组目标检测对比可从仓库根目录运行：

```powershell
python train.py detection-matrix --dry-run
python train.py detection-matrix
```
