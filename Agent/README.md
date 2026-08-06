# 智能识别 Agent

`Agent` 是整套识别流程的总控层，对应流程图中的：

1. 输入图像与多模态对齐信息；
2. 图像质量评估、特征提取与增广建议；
3. 模态识别；
4. 场景分类；
5. 目标框选与目标分类；
6. “模态 + 场景 + 目标”的一致性推理；
7. 决策指令、专家路由、损失代理值和持续学习记忆输出。

面向汇报讲解的完整流程整理见 [EXPLANATION_FLOW.md](EXPLANATION_FLOW.md)。

## 模块对接关系

Agent 只做调度；可复用能力直接调用仓库既有模块：

| Agent 阶段 | 复用模块 |
|---|---|
| 质量指标 / 增广建议 | `image_processing.scene_runtime` |
| 手工特征 | `image_processing.feature_engineering.extract_one` |
| 场景 SVM | `scene_recognition.feature_infer.predict_scene_from_features` |
| 场景 CNN（可选） | `image_processing.scene_runtime.predict_scene_cnn` |
| YOLO 标签解析 | `scene_recognition.detector_module.boxes` |
| 目标裁剪分类 | `scene_recognition.target_classifier_module.infer` |
| 场景决策策略基表 | `image_processing.scene_runtime.DEFAULT_POLICY` |

一致性推理、记忆、损失代理仍由 Agent 自身完成。

## 单图运行

```powershell
python -m Agent.cli infer `
  --image path\to\image.png `
  --sensor ir `
  --output Agent\runs\demo_report.json
```

真实模型是可选项。有权重时可以接入：

```powershell
python -m Agent.cli infer `
  --image path\to\image.png `
  --sensor sar `
  --scene-model scene_recognition\runs\feature_baseline\scene_feature_svm.joblib `
  --scene-metadata scene_recognition\runs\feature_baseline\model_metadata.json `
  --detector-model scene_recognition\detector_module\runs\yolov8n_baseline_v1\weights\submission_map50.pt `
  --target-checkpoint scene_recognition\target_classifier_module\runs\resnet18_target_baseline_none\best.pt
```

可选 CNN 场景后端与质量标定：

```powershell
python -m Agent.cli infer `
  --image path\to\image.png `
  --sensor ir `
  --scene-cnn-checkpoint path\to\scene_cnn.pt `
  --calibration path\to\calibration.json
```

如果没有检测权重，Agent 会尝试读取图片同名 YOLO 标签，例如 `xxx.png` 对应 `xxx.txt`。

## 批量运行

CSV 至少包含 `image` 或 `image_path` 列，可选 `sensor` 列：

```powershell
python -m Agent.cli batch `
  --manifest samples.csv `
  --output-dir Agent\runs\batch_demo
```

## 数据准备 / 训练桥接

下列子命令会转发到 `python train.py ...`，方便从 Agent 入口使用仓库流水线：

```powershell
python -m Agent.cli prepare-scene -- --dataset data\datasets_r1_base_train --output image_processing\artifacts
python -m Agent.cli extract-scene-features -- --index image_processing\artifacts\scene_index.csv --output image_processing\artifacts\scene_features.csv
python -m Agent.cli evaluate-scene -- --features image_processing\artifacts\scene_features.csv --output image_processing\runs\feature_eval_report
python -m Agent.cli prepare-crops -- ...
python -m Agent.cli prepare-detection -- ...
python -m Agent.cli train-detector -- ...
python -m Agent.cli train-target -- ...
```

`--` 之后的参数原样转发给对应 `train.py` 命令。

## 人工反馈与持续学习记忆

```powershell
python -m Agent.cli feedback `
  --image path\to\image.png `
  --scene forest `
  --modality ir `
  --targets soldier,tank `
  --note "人工复核后修正"
```

反馈会写入 `Agent/artifacts/agent_memory.jsonl`。

## 损失公式辅助

```powershell
python -m Agent.cli loss `
  --l-box 0.12 `
  --l-cls 0.18 `
  --l-dfl 0.04 `
  --l-detail 0.03 `
  --l-scene 0.10 `
  --l-proto 0.05 `
  --l-moti 0.08
```

对应公式：

```text
L = L_box + L_cls + L_dfl
  + lambda_detail * L_detail
  + lambda_scene * L_scene
  + lambda_proto * L_proto
  + lambda_moti * L_moti
```
