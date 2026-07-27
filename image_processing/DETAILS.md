# 场景识别特征工程模块

本目录已经基于实际数据集完成了数据审计、手工特征提取、特征消融、特征筛选、传统分类器验证和单图推理。

## 你负责的工作是什么

你的核心工作不是“训练一个大模型”，而是回答三个工程问题：

1. air、sea、urban、forest 四类场景在图像上到底有什么可量化差异？
2. 哪些特征对场景识别有效，哪些特征只是冗余或干扰？
3. 如何把有效特征整理成稳定、轻量、可解释的输入，交给场景分类器或后续决策模块？

当前实现提取三组共 68 个视觉特征：

- 灰度分布 21 个：均值、方差、分位数、动态范围、熵、偏度、峰度等。
- 纹理结构 41 个：梯度、边缘密度、拉普拉斯、局部方差、LBP、GLCM等。
- 频域特征 6 个：低/中/高频能量、频谱熵、加权频率半径等。

## 已获得的真实结果

数据划分不是随机打散，而是在每个“传感器×场景”子组内按文件序列做前70%/中15%/后15%划分，减少相邻帧泄漏。

| 特征集合 | 特征数 | 测试准确率 | 测试Macro-F1 |
|---|---:|---:|---:|
| 灰度分布 | 21 | 76.58% | 75.22% |
| 纹理 | 41 | 91.89% | 91.41% |
| 频域 | 6 | 72.07% | 68.11% |
| LBP | 17 | 92.79% | 92.33% |
| 纹理+频域 | 47 | 92.79% | 92.47% |
| 全部视觉特征 | 68 | 90.99% | 90.60% |
| ANOVA筛选后的30特征 | 30 | 92.79% | 92.25% |

结论：纹理是本数据集最有效的场景区分信息，尤其是LBP；简单堆叠全部特征会产生冗余。模型应优先使用筛选后的纹理特征，再少量补充灰度动态范围和频谱熵。

## 目录说明

- `analyze_and_prepare.py`：检查文件、图像、YOLO标签、重复项和候选IR/SAR配对，生成严格划分。
- `feature_engineering.py`：提取68个特征并完成消融、筛选和分类验证。
- `feature_infer.py`：加载筛选后的特征模型，对单幅图像输出场景概率。
- `scene_runtime.py`：图像质量特征与初版策略决策输出。
- `artifacts/scene_index.csv`：750幅图像的场景索引和划分。
- `artifacts/scene_features.csv`：完整手工特征表。
- `runs/feature_baseline/scene_feature_svm.joblib`：可直接推理的30特征SVM模型。
- `runs/feature_baseline/model_metadata.json`：模型输入字段、筛选结果和测试指标。
- `runs/feature_baseline/feature_ablation.csv`：各组特征消融结果。
- `runs/feature_baseline/scene_feature_signatures.json`：各传感器、各场景最明显的特征差异。
- `runs/feature_baseline/top_feature_importance.png`：特征重要性图。
- `runs/feature_baseline/pca_scene_sensor.png`：场景与模态的特征空间分布图。
- `特征工程实测报告.md`：可直接用于团队汇总的文字结果。

## 复现命令

在当前项目根目录执行：

```powershell
python .\image_processing\analyze_and_prepare.py `
  --dataset "$env:SCENE_DATASET" `
  --output .\image_processing\artifacts

python .\image_processing\feature_engineering.py extract `
  --index .\image_processing\artifacts\scene_index.csv `
  --output .\image_processing\artifacts\scene_features.csv

python .\image_processing\feature_engineering.py evaluate `
  --features .\image_processing\artifacts\scene_features.csv `
  --output .\image_processing\runs\feature_baseline
```

单图推理：

```powershell
python .\scene_recognition\feature_infer.py `
  --image '待识别图像.png' `
  --model .\image_processing\runs\feature_baseline\scene_feature_svm.joblib `
  --metadata .\image_processing\runs\feature_baseline\model_metadata.json
```

## 使用限制

- air场景只有红外样本，没有SAR样本，不能声称模型已验证“SAR空中场景”。
- 同编号IR/SAR抽样看起来是同一场景，但存在旋转、平移和尺度变化，不能直接视为像素级配准。
- 当前阈值和分类结果只对已给基础数据集负责，后续新增场景或真实数据需重新校准。
- `.joblib`模型只能从可信本地来源加载。
