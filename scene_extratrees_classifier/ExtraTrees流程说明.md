# ExtraTrees 场景识别流程

## 1. 处理流程

```text
输入图片
  ↓
转灰度图并缩放到 128×128
  ↓
提取 30 维特征
（灰度、梯度、边缘、局部标准差、LBP、GLCM、频谱熵）
  ↓
ExtraTrees 多分类训练
  ↓
输出 air / sea / urban / forest 及各类概率
```

## 2. 训练

场景名在文件夹名或文件名中时：

```powershell
python .\train_extratrees_classifier.py `
  --data-root "D:\data\datasets_r1_base_train"
```

使用 CSV 场景标签时：

```powershell
python .\train_extratrees_classifier.py `
  --data-root "D:\data\datasets_r1_base_train" `
  --labels-csv "D:\data\datasets_r1_base_train\scene_labels.csv"
```

快速运行、不搜索参数：

```powershell
python .\train_extratrees_classifier.py `
  --data-root "D:\data\datasets_r1_base_train" `
  --no-search
```

## 3. 单图识别

```powershell
python .\feature_infer_extratrees.py --image "待识别图片.png"
```

模型默认保存在：

```text
runs/extratrees/scene_feature_extratrees.joblib
```

同时会生成测试结果、混淆矩阵和特征重要性文件。
