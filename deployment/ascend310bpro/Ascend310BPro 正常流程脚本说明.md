# Ascend310BPro 正常流程脚本说明

这份说明只覆盖正常运行最常用的脚本。`ascend310bpro` 里还有一些底层
Python 文件，它们通常由 `.sh` 包装脚本自动调用，不需要手动执行。

## 0. 进入环境

每次在微机上运行前，先执行：

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r deployment/ascend310bpro/requirements-runtime.txt
```

检查环境：

```bash
bash deployment/ascend310bpro/check_env.sh
```

这个脚本会检查 Python、`atc`、`npu-smi`、`acl`、`numpy`、`Pillow`、
`attr`、`tornado`、`absl` 等。若 ATC 报 `np.float_ was removed` 或
`No module named 'attr'`，重新执行上面的 `pip install -r ...`。

## 1. 最推荐的正式测试流程

如果目标是按测试要求跑基础模型、ER500 增量模型、DER500 增量模型，并生成
提交格式结果，直接用：

```bash
bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 615738 \
  --no-save-images
```

它会自动执行两组策略：

```text
er500: base_before + base_after + increment_after
der500: base_before + base_after + increment_after
```

默认使用这些权重：

```text
基础四类模型: deployment/ascend310bpro/models/r1-four-class-detector-easy-640.pt
ER500 增量模型: deployment/ascend310bpro/models/class-il-er500-stage06-best.pt
DER500 增量模型: deployment/ascend310bpro/models/class-il-der500-stage06-best.pt
```

默认测试集：

```text
基础测试集: data/testdata/base_test_r1
增量测试集: data/testdata/inc_test_r2
```

主要输出：

```text
outputs/ascend310bpro_release_tzb/exports/
outputs/ascend310bpro_release_tzb/er500/
outputs/ascend310bpro_release_tzb/der500/
```

`exports/` 里缓存 `.onnx` 和 `.om`。第一次会比较慢，因为要 `.pt -> .onnx`
再由 ATC 编译 `.onnx -> .om`；后续只要不加 `--force-export` 和
`--force-convert`，就会复用缓存。

常用可选参数：

```bash
# 保存带框图片，便于检查识别效果
bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 615738

# 指定自己的测试集路径
bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 615738 \
  --base-test /path/to/base_test_r1 \
  --increment-test /path/to/inc_test_r2 \
  --no-save-images

# 强制重新导出 ONNX 和重新转换 OM，一般不要加
bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 615738 \
  --force-export \
  --force-convert
```

## 2. 单张图片演示流程

现场演示“输入一张图，输出识别效果”时，用：

```bash
IMG=$(find data/testdata/base_test_r1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | head -n 1)

bash deployment/ascend310bpro/run_single_image_demo.sh \
  --image "$IMG" \
  --strategy er500 \
  --soc-version Ascend310B4
```

可选策略：

```text
--strategy base   使用增量前四类基础模型
--strategy er500  使用 ER500 六类增量模型
--strategy der500 使用 DER500 六类增量模型
```

输出位置：

```text
outputs/ascend310bpro_demo/base/images/
outputs/ascend310bpro_demo/er500/images/
outputs/ascend310bpro_demo/der500/images/
```

每次运行会输出：

```text
outputs/ascend310bpro_demo/<strategy>/images/<输入图片名>.jpg
outputs/ascend310bpro_demo/<strategy>/predictions.jsonl
outputs/ascend310bpro_demo/<strategy>/summary.json
```

终端也会打印检测数量、类别、置信度和框坐标。正式演示时建议先跑一次完整
测试，让 `.om` 缓存生成好，这样单图演示会很快。

## 3. 原始数据集增广

如果只想把原始 YOLO/扁平数据集增广成标准 YOLO 数据集，用：

```bash
bash deployment/ascend310bpro/run_augment_yolo.sh \
  --data data/datasets_r1_base_train \
  --output outputs/ascend310bpro_full/augmented_dataset \
  --classes deployment/ascend310bpro/classes_6.txt
```

你的微机原始数据集 `data/datasets_r1_base_train` 是扁平结构也能处理。

输出：

```text
outputs/ascend310bpro_full/augmented_dataset/data.yaml
outputs/ascend310bpro_full/augmented_dataset/images/train
outputs/ascend310bpro_full/augmented_dataset/labels/train
outputs/ascend310bpro_full/augmented_dataset/augmentation_manifest.csv
outputs/ascend310bpro_full/augmented_dataset/augmentation_summary.json
```

常用参数：

```bash
# 已经增广过，直接复用
bash deployment/ascend310bpro/run_augment_yolo.sh \
  --data data/datasets_r1_base_train \
  --output outputs/ascend310bpro_full/augmented_dataset \
  --reuse-existing

# 确认要重新生成，旧目录会被移到 backup
bash deployment/ascend310bpro/run_augment_yolo.sh \
  --data data/datasets_r1_base_train \
  --output outputs/ascend310bpro_full/augmented_dataset \
  --rebuild-existing
```

## 4. 一键完整流程

如果想从原始数据集开始，完成增广、模型转换、推理和 Agent 报告，用：

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_full \
  --soc-version Ascend310B4 \
  --include-modality
```

如果已经有增广数据集，改成：

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_full \
  --reuse-augmented \
  --soc-version Ascend310B4 \
  --include-modality
```

常用阶段控制：

```text
--augment-only   只增广，不推理
--convert-only   只转换模型为 OM
--infer-only     只推理，跳过增广和显式转换
--report-only    只把已有 predictions.jsonl 生成 Agent 报告
--skip-report    推理后不生成 Agent 报告
```

一键流程默认是三模型路由方案，使用：

```text
deployment/ascend310bpro/route_config.yaml
deployment/ascend310bpro/models/01_scene_router_224.onnx
deployment/ascend310bpro/models/02_easy_detector_6class_640.onnx
deployment/ascend310bpro/models/03_hard_detector_3class_960.onnx
```

## 5. 只做模型转换

如果只想提前把 ONNX 转成 OM，避免正式推理时等待 ATC：

```bash
bash deployment/ascend310bpro/convert_models.sh \
  --soc-version Ascend310B4
```

这是三模型路由转换。它会读取 `route_config.yaml`。

如果是单个六分类模型：

```bash
bash deployment/ascend310bpro/convert_models.sh \
  --single \
  --model deployment/ascend310bpro/models/best.onnx \
  --soc-version Ascend310B4
```

说明：

```text
.pt 不能直接给 ATC，需要先导出 ONNX。
.onnx 第一次转 OM 较慢。
.om 文件存在时会自动复用。
不要随便加 --force-convert。
```

## 6. 只跑推理

三模型路由推理：

```bash
bash deployment/ascend310bpro/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_routed
```

输出：

```text
outputs/ascend310bpro_routed/summary.json
outputs/ascend310bpro_routed/predictions.jsonl
outputs/ascend310bpro_routed/images/
```

单模型推理：

```bash
bash deployment/ascend310bpro/run_best_6class_npu.sh \
  --model outputs/ascend310bpro_release_tzb/exports/class-il-er500-stage06-best_640x640.onnx \
  --input data/testdata/base_test_r1 \
  --classes deployment/ascend310bpro/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_best_er500 \
  --decode-mode auto
```

如果只要 JSON/TXT，不想保存图片：

```bash
bash deployment/ascend310bpro/run_best_6class_npu.sh \
  --model outputs/ascend310bpro_release_tzb/exports/class-il-er500-stage06-best_640x640.onnx \
  --input data/testdata/base_test_r1 \
  --classes deployment/ascend310bpro/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_best_er500 \
  --decode-mode auto \
  --no-save-images
```

## 7. 只生成 Agent 报告

如果推理已经完成，只想把 `predictions.jsonl` 生成报告、CSV 和 memory：

```bash
bash deployment/ascend310bpro/run_agent_reports.sh \
  --predictions outputs/ascend310bpro_routed/predictions.jsonl \
  --summary outputs/ascend310bpro_routed/summary.json \
  --output-dir outputs/ascend310bpro_routed/agent_reports \
  --memory outputs/ascend310bpro_routed/agent_memory.jsonl \
  --rewrite-memory \
  --include-modality
```

输出：

```text
agent_reports/reports/*.json
agent_reports/batch_summary.csv
agent_reports/agent_summary.json
agent_memory.jsonl
```

## 8. 按测试要求跑某一个增量策略

`run_release_weight_tzb_tests.sh` 会同时跑 ER500 和 DER500。如果只想跑其中
一个策略，用底层脚本：

```bash
bash deployment/ascend310bpro/run_tzb_board_tests.sh \
  --base-test data/testdata/base_test_r1 \
  --increment-test data/testdata/inc_test_r2 \
  --before-model deployment/ascend310bpro/models/r1-four-class-detector-easy-640.pt \
  --after-model deployment/ascend310bpro/models/class-il-er500-stage06-best.pt \
  --workspace outputs/ascend310bpro_tzb_board/er500 \
  --export-dir outputs/ascend310bpro_release_tzb/exports \
  --soc-version Ascend310B4 \
  --team-id 615738 \
  --strategy-label er500 \
  --no-save-images
```

这个脚本会生成三份推理结果：

```text
base_before      增量前模型跑基础测试集
base_after       增量后模型跑基础测试集
increment_after  增量后模型跑增量测试集
```

并整理成测试要求中的 TXT 文件夹。

## 9. 本地评估 mAP/KRR/New-mAP

如果手里有带标签测试集，可以用：

```bash
bash deployment/ascend310bpro/run_evaluate_tzb.sh \
  --base-data data/testdata/base_test_r1 \
  --new-data data/testdata/inc_test_r2 \
  --before-predictions outputs/ascend310bpro_tzb_board/er500/base_before/predictions.jsonl \
  --after-base-predictions outputs/ascend310bpro_tzb_board/er500/base_after/predictions.jsonl \
  --after-new-predictions outputs/ascend310bpro_tzb_board/er500/increment_after/predictions.jsonl \
  --fps-summary outputs/ascend310bpro_tzb_board/er500/increment_after/summary.json \
  --output outputs/ascend310bpro_tzb_board/er500/tzb_metrics.json
```

注意：如果测试集没有标签，这个评估脚本不能算真实 mAP，只能保留推理结果和
FPS 证明材料。

## 10. 可选：微机上做增量训练

微机上训练可以跑，但会慢；正常比赛推理不需要现场训练。训练必须从 `.pt` 或
`.yaml` 开始，不能用 `.onnx` 或 `.om`。

冒烟测试：

```bash
bash deployment/ascend310bpro/run_class_il_training.sh \
  --data outputs/ascend310bpro_full/augmented_dataset/data.yaml \
  --prepared outputs/ascend310bpro_full/class_il_prepared \
  --initial-model deployment/ascend310bpro/models/yolo26n.pt \
  --output-root outputs/ascend310bpro_full/runs \
  --method der \
  --device cpu \
  --batch-size 2 \
  --workers 0 \
  --smoke-test
```

完整 ER 和 DER：

```bash
bash deployment/ascend310bpro/run_class_il_training.sh \
  --data outputs/ascend310bpro_full/augmented_dataset/data.yaml \
  --prepared outputs/ascend310bpro_full/class_il_prepared \
  --initial-model deployment/ascend310bpro/models/yolo26n.pt \
  --output-root outputs/ascend310bpro_full/runs \
  --method both \
  --device cpu \
  --batch-size 2 \
  --workers 0 \
  --epochs 30
```

如果已经有四类基础模型，想从它继续训练到六类：

```bash
bash deployment/ascend310bpro/run_increment_from_before.sh \
  --base-data outputs/ascend310bpro_full/augmented_dataset/data.yaml \
  --increment-data /path/to/six_class_increment/data.yaml \
  --before-model deployment/ascend310bpro/models/r1-four-class-detector-easy-640.pt \
  --workspace outputs/ascend310bpro_increment \
  --method der \
  --device cpu \
  --batch-size 2 \
  --workers 0
```

## 11. 可选：PC 上训练基础模型

这个脚本给 Windows 训练机用，不在微机上跑：

```powershell
Set-Location "D:\000zzjCodes\NCEPUWorkspace\TiaoZhanBeiWorkspace\ChallengeCup"
conda activate zzjDEREnv

powershell -ExecutionPolicy Bypass -File deployment\ascend310bpro\train_before_model_pc.ps1 `
  -DataRoot data\datasets_r1_base_train `
  -OutputRoot outputs\ascend310bpro_before_pc `
  -InitialModel deployment\ascend310bpro\models\yolo26n.pt `
  -Epochs 5 `
  -BatchSize 8 `
  -Device 0
```

用于在电脑上快速训练或复现增量前四类模型。

## 12. 脚本关系速查

正式测试一键入口：

```text
run_release_weight_tzb_tests.sh
  -> run_tzb_board_tests.sh
     -> export_yolo_pt_to_onnx.py
     -> run_best_6class_npu.sh
        -> infer_best_6class_npu.py
           -> ATC ONNX-to-OM conversion/reuse
           -> Ascend ACL NPU inference
     -> run_agent_reports.sh
     -> tzb_format_results.py
```

完整项目一键入口：

```text
run_full_pipeline.sh
  -> run_augment_yolo.sh
  -> convert_models.sh
  -> run_routed_infer.sh or run_best_6class_npu.sh
  -> run_agent_reports.sh
```

现场演示入口：

```text
run_single_image_demo.sh
  -> export_yolo_pt_to_onnx.py if input is .pt
  -> run_best_6class_npu.sh
  -> outputs/ascend310bpro_demo/<strategy>/images/*.jpg
```

## 13. 最容易踩的点

1. `.pt` 是训练权重，不能直接给 NPU 跑；脚本会先导出 ONNX，再转 OM。
2. `.om` 是昇腾推理模型，不能继续训练。
3. ATC 第一次慢是正常的，后续会复用 `.om`。
4. 不要随便删除 `outputs/ascend310bpro_release_tzb/exports`，里面是缓存。
5. 不要随便加 `--force-convert`，除非模型或输入尺寸真的换了。
6. 单模型 Ultralytics 导出的 ONNX 建议加 `--decode-mode auto`。
7. 四类基础模型用 `classes_base4.txt`，六类增量模型用 `classes_6.txt`。
8. 正式测试只要结果文件时加 `--no-save-images`，演示可视化时不要加。
