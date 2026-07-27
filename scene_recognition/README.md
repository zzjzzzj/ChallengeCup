# 场景识别模块

本目录负责 air、sea、urban、forest 四类场景模型的训练、评估、推理和可视化。

当前包含两条路线：

- 手工特征路线：StandardScaler + ANOVA特征筛选 + RBF-SVM；
- 图像模型路线：直接读取场景图像训练分类模型。

查看命令：

```powershell
python -m scene_recognition.cli --help
```

手工特征模型训练：

```powershell
python -m scene_recognition.cli train-features `
  --features scene_module/artifacts/scene_features.csv `
  --output scene_module/runs/feature_eval_report
```

单图推理：

```powershell
python -m scene_recognition.cli infer `
  --image "D:\samples\example.png" `
  --model scene_module/runs/feature_eval_report/scene_feature_svm.joblib `
  --metadata scene_module/runs/feature_eval_report/model_metadata.json
```

启动本地分析台：

```powershell
python -m scene_recognition.cli dashboard
```

底层实现暂时保留在 `scene_module/`，以兼容既有实验命令和测试。
