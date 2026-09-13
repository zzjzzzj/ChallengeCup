# 昇腾 310B 部署说明

本目录用于为 aarch64 架构的昇腾 310B 设备构建轻量运行包。设  备端使用
CANN ACL Python 调用 OM 模型推理，不依赖 PyTorch、torchvision 或 Ultralytics。

## 在训练或导出设备上构建运行包

请在项目根目录、且可导入 `torch`、`ultralytics`、`onnx` 和 `PyYAML` 的 Python
环境中执行。

```bash
python train.py ascend310b-package \
  --checkpoint scene_recognition/detector_module/runs/ascend_yolov8n_960_aug_pretrained/weights/best.pt \
  --data scene_recognition/detector_module/artifacts/comparison_dataset/data_aug.yaml \
  --image-size 960 \
  --output dist/ascend310b_yolov8n_960 \
  --archive \
  --force
```

如果已经有静态 batch=1 的 ONNX 文件，可直接打包：

```bash
python train.py ascend310b-package \
  --onnx path/to/detector_yolov8n_bs1.onnx \
  --classes data/datasets_r1_base_train/classes.txt \
  --image-size 960 \
  --output dist/ascend310b_yolov8n_960 \
  --archive \
  --force
```

将生成的文件夹或 zip 压缩包复制到昇腾设备即可。

## 可选：板端训练前的数据增广

本节适用于在微型计算机上保留完整项目代码的场景，不适用于仅含推理文件的运行包。
代码使用项目内的数据集根目录，不依赖硬编码的 Windows 路径。

先安装轻量增广依赖。若要进行 YOLO 训练，还需针对 aarch64 环境另行安装
torch/torchvision。

```bash
python3 -m pip install -r deployment/ascend310b/requirements-training.txt
```

生成增广后的 YOLO 训练集：

```bash
python3 train.py ascend310b-augment \
  --dataset-root data/datasets_r1_base_train \
  --output outputs/datasets_r1_base_train_augmented \
  --include-original \
  --classes data/datasets_r1_base_train/classes.txt
```

`--dataset-root` 同时支持标准 YOLO 目录结构
`images[/train] + labels[/train]`，以及本项目板端使用的扁平目录结构：图片、同名
`.txt` 标签和 `classes.txt` 位于同一目录。

输出目录包含 `images/`、`labels/`、`classes.txt`、
`augmentation_manifest.csv`、`augmentation_summary.json` 和 `data.yaml`。
若未传入独立验证集，生成的 `data.yaml` 会将 `val` 指向增广训练图片，便于完成
连通性测试，但该结果不能作为独立精度指标。

如果有独立验证集，请在生成 YAML 时一并传入：

```bash
python3 train.py ascend310b-augment \
  --dataset-root data/datasets_r1_base_train \
  --val-root data/datasets_r1_base_val \
  --output outputs/datasets_r1_base_train_augmented \
  --include-original \
  --classes data/datasets_r1_base_train/classes.txt
```

随后可使用生成的 YAML 训练：

```bash
python3 train.py yolo \
  --data outputs/datasets_r1_base_train_augmented/data.yaml \
  --image-size 960 \
  --batch-size 4 \
  --workers 2 \
  --device cpu \
  --name ascend310b_augmented_yolov8n_960
```

也可以用一个命令完成增广与训练。若增广集的 `data.yaml` 已存在，默认会复用；只有
传入 `--force-augment` 时才重新构建。

```bash
bash deployment/ascend310b/run_train_with_aug.sh \
  --dataset-root data/datasets_r1_base_train \
  --output outputs/datasets_r1_base_train_augmented \
  --classes data/datasets_r1_base_train/classes.txt \
  --image-size 960 \
  --batch-size 4 \
  --workers 2 \
  --device cpu \
  --name ascend310b_augmented_yolov8n_960
```

## 端到端流程：数据增广、已训练模型推理和结果输出

模型已经训练并导出后，使用下列统一入口即可依次完成指定的数据增广、模型推理与
结果归档。该流程**不会再次训练模型**。

ONNX CPU 连通性测试示例：

```bash
python3 train.py ascend310b-pipeline \
  --dataset-root data/datasets_r1_base_train \
  --classes data/datasets_r1_base_train/classes.txt \
  --model models/detector_yolov8n_bs1.onnx \
  --backend onnx \
  --output-dir outputs/ascend310b_pipeline_onnx
```

昇腾 OM 推理示例：可直接传入已生成的 `.om` 模型；也可以传入 ONNX 文件并指定
`--backend om`，流程会复用已有的转换逻辑。

```bash
bash deployment/ascend310b/run_end_to_end.sh \
  --dataset-root data/datasets_r1_base_train \
  --classes data/datasets_r1_base_train/classes.txt \
  --model models/detector_yolov8n_960_bs1.om \
  --backend om \
  --metadata models/package_metadata.json \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310b_pipeline_om
```

未传入 `--infer-input` 时，流程会对本次生成的
`augmented_dataset/images` 执行推理。若要在独立的待测图像或留出图像上推理，请传入
`--infer-input path/to/images`；数据增广仍会正常执行并归档在同一个输出目录中。
每次的 `--output-dir` 必须是不存在或为空的目录，以避免静默覆盖历史结果。

输出目录包含：

- `augmented_dataset/`：成对的增广图片与标签、`data.yaml`、增广清单和增广摘要；
- `inference/predictions.json`：模型预测结果；
- `inference/annotated_images/`：绘制检测框、类别和置信度的图片；
- `inference/result_summary.csv`：按图片汇总的目标类别数、目标总数、各类数量、置信度
  和固定中文描述，可直接用 Excel 打开；
- `inference/runtime_metadata.json`：未传入模型元数据时自动生成的类别与输入元数据；
- `pipeline_summary.json`：本次运行的模型、后端、增广信息、输出路径、图片数量和检测
  数量汇总。

`result_summary.csv` 由项目根目录的 `result_formatter.py` 生成，不使用大模型或聊天
接口。它按检测类别统计数量，并保留每个检测框的置信度，例如“检测到 2 类目标，共
5 个：轮船/舰船 3 个（置信度：0.94、0.91、0.88）”。当前 310B ONNX/OM 推理链只输出
目标检测结果，尚未接入场景分类模型，因此该 CSV 的 `scene` 列会为空，中文描述会标明
“图像场景分类：未提供”。完整 Agent 流程接入场景分类后会填充该字段。

## 必要时在微型计算机上进行增量训练

昇腾 310B 板卡的可靠交付路径是 OM/ACL 推理。若项目要求在微型计算机上进行增量
训练，建议使用 CPU PyTorch 更新权重，并使用 NPU 进行模型转换和推理验证。

先检查实际运行的 Python 环境和训练依赖是否可用：

```bash
bash deployment/ascend310b/run_probe_training_env.sh
```

若缺少 `torch` 或 `ultralytics`，请安装到实际启动训练的同一环境中。先从适合板卡的
CPU/aarch64 wheel 来源安装 torch/torchvision，再安装项目训练依赖：

```bash
python -m pip install -U pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deployment/ascend310b/requirements-training.txt
```

板端增量微调应先做连通性测试，不建议直接执行完整 960 像素训练：

```bash
python train.py continual-yolo \
  --data scene_recognition/detector_module/artifacts/continual_r2/data_replay.yaml \
  --base-model models/base_4class.pt \
  --strategy replay \
  --output scene_recognition/detector_module/runs/micro_continual_smoke \
  --epochs 2 \
  --patience 999 \
  --image-size 512 \
  --batch-size 1 \
  --workers 0 \
  --device cpu \
  --freeze 10 \
  --no-amp \
  --no-plots \
  --no-builtin-aug
```

连通性测试通过后，每次只增加一项预算：先增加 `--epochs`，再增加 `--image-size`，
最后再减少或取消 `--freeze`。实际板端运行建议从
`--image-size 640 --batch-size 1 --workers 0` 开始。

对于六阶段 Class-IL 训练器，建议优先使用 ER；DER 训练期间还会额外加载教师模型：

```bash
python train.py class-il-yolo \
  --prepared scene_recognition/detector_module/artifacts/class_incremental \
  --initial-model models/yolov8n.pt \
  --method er \
  --buffer-size 200 \
  --output scene_recognition/detector_module/runs/micro_class_il_er \
  --epochs 2 \
  --patience 999 \
  --image-size 512 \
  --batch-size 1 \
  --workers 0 \
  --device cpu \
  --freeze 10 \
  --no-amp \
  --no-plots \
  --no-builtin-aug \
  --stop-after-stage 1
```

当且仅当已安装 `torch_npu` 时，`--device npu:0` 可用于实验；它并非昇腾 310B 的
推荐交付路径。可靠的微型计算机流程仍是 CPU 训练加 NPU OM 推理。

## 先在 CPU 上运行 ONNX

ONNX CPU 通常是最容易在板端完成的连通性测试，建议先于 OM 推理执行：

```bash
cd ascend310b_yolov8n_960
python3 -m pip install -r requirements-onnx-cpu.txt

bash run_onnx_cpu.sh \
  --model detector_yolov8n_bs1.onnx \
  --image demo.png \
  --metadata package_metadata.json \
  --output result_onnx_cpu.json \
  --save-image outputs_onnx_cpu
```

该步骤会验证微型计算机上的完整预处理、后处理、JSON 输出和可视化流程。

## 在昇腾 310B 设备上转换 OM 模型

```bash
cd ascend310b_yolov8n_960
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r requirements-runtime.txt

npu-smi info
atc --list_soc_version
export SOC_VERSION=Ascend310B4
bash convert_onnx_to_om.sh detector_yolov8n_bs1.onnx detector_yolov8n_960_bs1
```

请将 `Ascend310B4` 替换为板卡实际支持的值。

## 运行单模型推理

```bash
bash run_infer.sh \
  --model detector_yolov8n_bs1.onnx \
  --image demo.png \
  --soc-version Ascend310B4 \
  --metadata package_metadata.json \
  --output result.json \
  --save-image outputs
```

对于图片目录，将该目录传入 `--image` 即可；JSON 会为每张图片写入一个结果对象。
若 `--model` 为 ONNX 文件，脚本会先使用 ATC 转换为 OM，并在后续运行时复用已缓存的
OM 文件；也可直接传入已有 `.om` 文件。

## 运行双模型级联推理

两个 OM 模型均已在板端可用后，使用下列命令：

```bash
bash run_cascade_npu.sh \
  --input images \
  --main-model models/six_class_960.onnx \
  --expert-model models/soldier_expert_1024x832.onnx \
  --main-classes cascade_classes_6.txt \
  --expert-class soldier \
  --soc-version Ascend310B4 \
  --output-dir cascade_outputs
```

脚本会在需要时先将 ONNX 转换为 OM。默认策略会对每张图片运行 1024x832 专家模型，
将专家类别与六分类模型结果融合，写入 `summary.json`、`predictions.jsonl`，并将标注图
保存至 `cascade_outputs/images`。

如果优先考虑速度而非召回率，仅在六分类模型漏检专家类别或该类别置信度较低时触发专家：

```bash
bash run_cascade_npu.sh \
  --input images \
  --main-model models/six_class_960.onnx \
  --expert-model models/soldier_expert_1024x832.onnx \
  --soc-version Ascend310B4 \
  --expert-strategy missing-or-low-confidence \
  --expert-trigger-conf 0.45 \
  --output-dir cascade_outputs_fast
```

若 OM 输出为 `300x6` 的紧凑后处理格式，例如
`x1,y1,x2,y2,conf,class_id`，保持默认参数即可。若输出为原始 YOLO 头，请传入
`--main-output-mode raw --expert-output-mode raw`。
若仅专家模型采用不同的紧凑格式，主模型参数无需调整，只需设置
`--expert-nms-format`，例如 `--expert-nms-format xywh-conf-class`。

也可以直接传入 ONNX 模型。脚本会调用 `atc`，缓存转换得到的 OM 文件，然后继续执行
NPU 推理：

```bash
bash run_cascade_npu.sh \
  --input images \
  --main-model models/detector_6class_960.onnx \
  --expert-model models/soldier_expert_1024x832.onnx \
  --main-classes cascade_classes_6.txt \
  --expert-class soldier \
  --soc-version Ascend310B4 \
  --output-dir cascade_outputs
```

生成的 OM 文件名会包含输入尺寸和 soc 版本，例如
`detector_6class_960_960x960_Ascend310B4.om`。如果该 OM 文件已存在，会被直接复用；
传入 `--force-convert` 可重新转换。

ATC 使用 NCHW 输入形状。未传入宽高时，脚本会先读取 ONNX 输入形状；若读取失败，
会根据模型文件名推断，例如 `soldier_legacy4_1120x896.onnx`。该模型对应的 ATC 形状为
`images:1,3,896,1120`。

也可以显式覆盖输入尺寸：

```bash
bash run_cascade_npu.sh \
  --input images \
  --main-model models/detector_6class_960.onnx \
  --expert-model models/soldier_legacy4_1120x896.onnx \
  --expert-classes cascade_classes_6.txt \
  --expert-width 1120 \
  --expert-height 896 \
  --soc-version Ascend310B4 \
  --output-dir cascade_outputs
```

真正的单类别专家模型不需要 `--expert-classes`。对于
`soldier_legacy4_1120x896.onnx` 这类历史多类别专家模型，必须传入其类别文件，级联流程
才会保留指定 `--expert-class` 的检测结果，而不是把专家模型的全部检测都视为 soldier。

## 使用说明

- 运行时脚本兼容 Python 3.9。
- 完整项目仍包含训练和导出逻辑，训练或导出设备可能需要更高版本的 Python 和 PyTorch。
- OM 模型使用静态 batch=1、NCHW、float32 输入。
- NMS 在 `infer_yolov8_om.py` 中的 CPU 侧执行。
