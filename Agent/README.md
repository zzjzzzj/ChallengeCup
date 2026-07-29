# 智能识别 Agent

`Agent` 是整套识别流程的总控层，对应流程图中的：

1. 输入图像与多模态对齐信息；
2. 图像质量评估、特征提取与增广建议；
3. 模态识别；
4. 场景分类；
5. 目标框选与目标分类；
6. “模态 + 场景 + 目标”的一致性推理；
7. 决策指令、专家路由、损失代理值和持续学习记忆输出。

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

如果没有检测权重，Agent 会尝试读取图片同名 YOLO 标签，例如 `xxx.png` 对应 `xxx.txt`。这能让流程在训练权重缺失时仍然演示“框选目标 -> 目标类别 -> 组合约束”的完整链路。

## 批量运行

CSV 至少包含 `image` 或 `image_path` 列，可选 `sensor` 列：

```powershell
python -m Agent.cli batch `
  --manifest samples.csv `
  --output-dir Agent\runs\batch_demo
```

## 人工反馈与持续学习记忆

```powershell
python -m Agent.cli feedback `
  --image path\to\image.png `
  --scene forest `
  --modality ir `
  --targets soldier,tank `
  --note "人工复核后修正"
```

反馈会写入 `Agent/artifacts/agent_memory.jsonl`。后续可把这些记录作为经验回放、类别原型或增量训练清单的来源。

## 损失公式辅助

流程图中的损失可用下面命令汇总：

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
