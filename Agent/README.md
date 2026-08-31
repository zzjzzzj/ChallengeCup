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

## 固定规则结果描述与 CSV 输出

Agent 不调用大模型或聊天接口生成结果说明。场景分类和目标检测完成后，
`result_formatter.py` 会使用固定脚本按目标类别分组，统计类别数、每类数量、目标总数和
每个检测框的置信度，并生成中文描述。例如：

```text
图像场景分类：海洋。检测到 2 类目标，共 5 个：
轮船/舰船 3 个（置信度：0.94、0.91、0.88）；巡逻艇 2 个（置信度：0.86、0.82）。
```

未检出目标时会输出“图像场景分类：海洋。未检测到目标。”。图像模态会保留在 JSON
报告和批量 CSV 中；默认不写入这段面向用户的描述。

单图 JSON 的顶层 `output_summary` 字段包含以下结构化内容：

- `target_type_count`：检测到的目标类别数；
- `target_total_count`：目标总数；
- `targets`：各类别的中文名、数量、全部置信度、最大/平均置信度；
- `target_details` 与 `description`：可直接展示或写入表格的中文文本。

批量命令继续生成每张图片的 JSON 与 `batch_summary.csv`，并在原有列基础上新增
`target_type_count`、`target_details`、`max_confidence` 和 `description`。CSV 使用 UTF-8
BOM 编码，可直接用 Windows Excel 打开。

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

对已经完成离线增广的 YOLO 数据集，可通过 Agent 桥接入口关闭二次在线增广：

```powershell
python -m Agent.cli train-detector -- `
  --data path\to\data.yaml `
  --model path\to\local_yolo_checkpoint.pt `
  --name offline_augmented_run `
  --no-builtin-aug
```

## r2 类增量学习

r2 在原四类后追加 `patrol_boat` 和 `armored_vehicle`，类别编号必须保持
`0..5` 不变。增量数据只在本机读取，生成的绝对路径清单、权重和报告均位于
Git 忽略目录。

```powershell
$env:R2_DATASET = "<本机 datasets_r2_inc_train 路径>"
$env:BASE_INDEX = "image_processing\artifacts\scene_index.csv"
$env:BASE_CHECKPOINT = "<本机四类基础模型 best.pt>"

# 正式协议：r2 全部用于训练；val/test 必须来自独立固定数据。
python -m Agent.cli prepare-continual -- `
  --increment-dataset "$env:R2_DATASET" `
  --base-index "$env:BASE_INDEX" `
  --output scene_recognition\detector_module\artifacts\continual_r2

# 回放基线；只使用本地 checkpoint，不会自动下载模型。
python -m Agent.cli train-continual -- `
  --data scene_recognition\detector_module\artifacts\continual_r2\data_replay.yaml `
  --base-model "$env:BASE_CHECKPOINT" `
  --strategy replay `
  --output scene_recognition\detector_module\runs\continual_r2_replay

# 固定 test 上计算 old-mAP、New-mAP、all-mAP 和 KRR。
python -m Agent.cli evaluate-continual -- `
  --data scene_recognition\detector_module\artifacts\continual_r2\data_replay.yaml `
  --before "$env:BASE_CHECKPOINT" `
  --after scene_recognition\detector_module\runs\continual_r2_replay\weights\best.pt `
  --output scene_recognition\detector_module\runs\continual_r2_replay\continual_evaluation.json
```

若只有 `inc_train`、没有独立新增类测试集，可显式使用
`--increment-val-ratio 0.1 --increment-test-ratio 0.1` 做本地冒烟测试；该划分来自训练注入，
其分数不得作为正式 New-mAP。完整设计与验收口径见
[`../docs/Agent持续学习完善说明.md`](../docs/Agent持续学习完善说明.md)。

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
