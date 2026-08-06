# Agent讲解流程

这份流程可以直接用于答辩或组会讲解。建议按“为什么需要Agent -> Agent如何调度 -> 单张图片输出了什么 -> 后续如何升级”的顺序讲。

## 1. 一句话定位

本项目的Agent不是单一识别模型，而是一个智能调度层。它把图像处理、模态识别、场景分类、目标检测、目标分类、场景目标一致性校验、持续学习记忆和端侧部署决策串成一个完整流程。

底层能力已对接仓库模块：

```text
质量/增强     -> image_processing.scene_runtime
手工特征      -> image_processing.feature_engineering
场景SVM       -> scene_recognition.feature_infer
场景CNN(可选) -> image_processing.scene_runtime.predict_scene_cnn
YOLO标签      -> scene_recognition.detector_module.boxes
目标裁剪分类  -> scene_recognition.target_classifier_module.infer
决策策略基表  -> image_processing.scene_runtime.DEFAULT_POLICY
```

对应输入输出：

```text
输入图像/多模态图像
    -> 图像质量评估与特征提取
    -> 模态识别
    -> 场景分类
    -> 目标定位与目标分类
    -> 场景+目标一致性推理
    -> 决策指令、损失代理、JSON报告
```

## 2. 数据进入系统

当前数据集路径：

```text
data/datasets_r1_base_train
```

数据格式是标准YOLO格式：

```text
xxx.png
xxx.txt
classes.txt
```

其中 `classes.txt` 的类别顺序是：

```text
soldier
small_aircraft
warship
tank
```

项目先通过 `scene-prepare` 生成统一索引（也可经 Agent CLI 桥接）：

```powershell
python train.py scene-prepare `
  --dataset data\datasets_r1_base_train `
  --output image_processing\artifacts

# 等价：
python -m Agent.cli prepare-scene -- --dataset data\datasets_r1_base_train --output image_processing\artifacts
```

这个步骤完成数据审计、场景/模态解析、train/val/test划分，并生成：

```text
image_processing/artifacts/scene_index.csv
```

## 3. 图像处理与增强建议

Agent 通过 `image_processing.scene_runtime.quality_metrics` 读取图像，计算：

- 灰度均值、对比度、动态范围；
- 清晰度、边缘强度；
- 高频噪声；
- 色彩强度；
- 图像宽高和模态上下文。

这些指标会被转换成 `contrast_level / clarity_level / noise_level` 等环境状态。

主增强建议来自 `scene_runtime.choose_enhancement`，例如：

- 低对比度：`contrast_stretch`；
- SAR高噪声：`speckle_denoise` / `sar_speckle_denoise`；
- 小目标边界弱：`mild_sharpen`；
- 森林/城市场景：额外追加 `small_rotation_and_flip` 用于增量训练数据增强。

讲解重点：这一步对应流程图左侧的“图像处理和增广”，并且为后面的动态决策提供环境状态向量。

## 4. 模态识别

模态识别输出三类概率：

```text
visible / ir / sar
```

优先级：

1. 如果命令行提供 `--sensor ir`，直接使用传感器提示；
2. 如果文件名中有 `ir` 或 `sar`，用文件名规则；
3. 否则根据颜色、噪声、对比度等图像统计特征进行判断。

讲解重点：实际部署时模态通常来自传感器通道；这里保留图像统计判断是为了增强系统鲁棒性。

## 5. 场景分类

场景类别为：

```text
air / sea / urban / forest
```

当前可以使用传统特征SVM模型：

```powershell
python train.py scene-extract `
  --index image_processing\artifacts\scene_index.csv `
  --output image_processing\artifacts\scene_features.csv

python train.py scene-evaluate `
  --features image_processing\artifacts\scene_features.csv `
  --output image_processing\runs\feature_eval_report
```

Agent调用时接入：

```powershell
--scene-model image_processing\runs\feature_eval_report\scene_feature_svm.joblib
--scene-metadata image_processing\runs\feature_eval_report\model_metadata.json
```

讲解重点：场景模型输出的是四类概率，不只是一个标签。后续还会和目标结果融合，得到最终场景。

## 6. 目标定位与目标分类

目标定位优先调用YOLO权重：

```powershell
--detector-model path\to\best.pt
```

如果没有训练好的YOLO权重，Agent会读取同名YOLO标签作为回退：

```text
sar_r1_base_forest_000016.png
sar_r1_base_forest_000016.txt
```

这样即使训练尚未完成，也能演示完整流程。

输出统一为：

```text
class_name
confidence
x_center / y_center / width / height
xyxy_norm
```

讲解重点：目标定位负责“在哪里”，目标分类负责“是什么”。后续接入ResNet18裁剪分类器后，可以对YOLO框再次确认类别。

## 7. 场景目标一致性推理

Agent内置了场景-目标组合规则：

```text
air    -> small_aircraft
sea    -> warship
urban  -> soldier / tank
forest -> soldier / tank
```

如果出现不合理组合，例如：

```text
sea + tank
air + warship
```

Agent会把它标为冲突，并提高 `L_proto` 或 `L_cls` 相关代理损失。

如果场景模型不确定，而目标结果很明确，Agent会让目标结果给场景投票。例如检测到多个 `warship` 时，会提高 `sea` 的最终概率。

讲解重点：这一步就是流程图里的“模态+场景+目标 => 唯一确定场景”。

## 8. 决策输出

最终报告中的 `decision` 字段包含：

- `detector_profile`：当前应使用哪种检测配置；
- `confidence_threshold`：当前帧检测阈值；
- `priority_classes`：重点关注类别；
- `sensor_weights`：多模态权重；
- `feature_weights`：特征权重；
- `expert_routing`：12个专家和跨模态适配器的路由；
- `model_management`：Ascend 310B部署策略。

例子：

```text
SAR + forest + soldier/tank
    -> forest_complex_background
    -> 激活 sar_soldier_expert、sar_tank_expert、cross_modal_adapter
```

## 9. 损失与二次评估能力

训练中的设计损失：

```text
L = L_box + L_cls + L_dfl
  + lambda_detail * L_detail
  + lambda_scene * L_scene
  + lambda_proto * L_proto
  + lambda_moti * L_moti
```

Agent推理时没有真实标签，因此输出的是运行时代理损失：

- 模态置信度低：`L_moti` 上升；
- 场景置信度低：`L_env` 上升；
- 检测置信度低：`L_box / L_cls` 上升；
- 图像质量差：`L_detail` 上升；
- 场景目标组合冲突：`L_proto` 上升。

讲解重点：代理损失不是训练反向传播的真实loss，而是在线质量评估和人工复核排序的依据。

## 10. 记忆与持续学习

Agent可以把每次推理写入：

```text
Agent/artifacts/agent_memory.jsonl
```

人工纠错命令：

```powershell
python -m Agent.cli feedback `
  --image data\datasets_r1_base_train\sar_r1_base_forest_000016.png `
  --scene forest `
  --modality sar `
  --targets soldier,tank `
  --note "人工复核后确认"
```

这些反馈后续可以用于：

- 经验回放；
- 类别原型更新；
- 增量训练清单；
- 错误案例分析。

## 11. 演示命令

单张图像：

```powershell
python -m Agent.cli infer `
  --image data\datasets_r1_base_train\sar_r1_base_forest_000016.png `
  --sensor sar `
  --scene-model image_processing\runs\feature_eval_report\scene_feature_svm.joblib `
  --scene-metadata image_processing\runs\feature_eval_report\model_metadata.json `
  --output Agent\runs\demo_report.json `
  --no-memory
```

小批量测试：

```powershell
python -m Agent.cli batch `
  --manifest image_processing\artifacts\scene_index.csv `
  --split test `
  --limit 10 `
  --output-dir Agent\runs\dataset_batch_test10 `
  --scene-model image_processing\runs\feature_eval_report\scene_feature_svm.joblib `
  --scene-metadata image_processing\runs\feature_eval_report\model_metadata.json `
  --no-memory
```

## 12. 汇报时的三句话总结

1. 我们用Agent把多模态识别拆成可解释、可替换、可联调的多个模块。
2. Agent不仅输出分类结果，还输出场景目标一致性、动态决策和损失代理，用于在线评估。
3. 后续只要替换YOLO/ResNet/端侧OM权重，整体流程和报告结构不需要改变。
