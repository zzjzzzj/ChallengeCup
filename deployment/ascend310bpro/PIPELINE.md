# Ascend 310B Pro Full Pipeline

This folder is a complete board-side wrapper for the current six-class
ChallengeCup deployment.

```text
original YOLO dataset
  -> offline train-set augmentation
  -> optional Class-IL incremental training
  -> ONNX to OM conversion
  -> Ascend 310B NPU inference
  -> Agent-style report outputs
  -> TZB test-result folder packaging
```

## Stage 1: Dataset Augmentation

Script:

```bash
bash deployment/ascend310bpro/run_augment_yolo.sh
```

It supports both standard YOLO `data.yaml` datasets and your flat board dataset
layout:

```text
data/datasets_r1_base_train/
  classes.txt
  ir_r1_base_air_000001.png
  ir_r1_base_air_000001.txt
  ...
```

For flat roots, it calls the corrected augmentation helper from
`deployment/ascend310b/augment_selected_yolo.py` and writes a normal YOLO
dataset:

```text
<workspace>/augmented_dataset/data.yaml
<workspace>/augmented_dataset/images/train
<workspace>/augmented_dataset/labels/train
<workspace>/augmented_dataset/augmentation_manifest.csv
<workspace>/augmented_dataset/augmentation_summary.json
```

The default class protocol is `classes_6.txt`:

```text
soldier
small_aircraft
warship
tank
patrol_boat
armored_vehicle
```

## Stage 2: Optional Class-IL Training

Script:

```bash
bash deployment/ascend310bpro/run_class_il_training.sh
```

It wraps:

```text
python train.py prepare-class-il
python train.py class-il-yolo
```

Training starts from a `.pt` or `.yaml` model. Do not use `.onnx` or `.om` as
the initial training model. The board-safe defaults are:

```text
device=cpu
batch-size=2
workers=0
method=both
```

Use `--device npu:0` only after the board's `torch_npu` training stack is
verified. Ascend 310B OM/NPU deployment is the reliable inference path; training
on the board can be slow or environment-dependent.

## Stage 3: ONNX To OM

Scripts:

```bash
bash deployment/ascend310bpro/convert_models.sh
bash deployment/ascend310bpro/run_best_6class_npu.sh --convert-only
```

The current default is the single six-class detector:

```text
deployment/ascend310bpro/models/best.onnx
```

The Python runtime auto-detects the static ONNX NCHW input size. This is
important for exported models such as `best.onnx` whose real input size may not
be `960x960`. Matching OM files are cached by model name, size, and SOC, for
example:

```text
best_320x320_Ascend310B4.om
```

Existing OM files are reused. Pass `--force-convert` only when you really want
to rebuild them.

## Stage 4: NPU Inference

Default single-model path:

```bash
bash deployment/ascend310bpro/run_best_6class_npu.sh \
  --model deployment/ascend310bpro/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bpro/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_best
```

Outputs:

```text
summary.json
predictions.jsonl
images/
```

The legacy routed path is still available with `--routed`:

```text
scene router -> easy detector for air/sea, hard detector for forest/urban
```

Use it only when you deliberately want the old three-model strategy.

## Stage 5: Agent-Style Outputs

Script:

```bash
bash deployment/ascend310bpro/run_agent_reports.sh
```

It reads `predictions.jsonl` and writes:

```text
agent_reports/reports/*.json
agent_reports/batch_summary.csv
agent_reports/agent_summary.json
agent_memory.jsonl
```

It adds the lightweight Agent functions that are useful on the board:

```text
modality inference from filename/default
scene inference or scene-router reuse
scene-target consistency check
final scene fusion
Chinese image summary
runtime proxy losses
decision/model-management metadata
feedback memory records
```

## Recommended Full Command

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_full \
  --reuse-augmented \
  --soc-version Ascend310B4 \
  --include-modality
```

When the augmented dataset does not exist yet, omit `--reuse-augmented`. When
you want to rebuild it from the original dataset, use `--rebuild-augmented`.

## Report Existing Predictions

If NPU inference has already finished:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --report-only \
  --predictions outputs/ascend310bpro_full/best_6class_infer/predictions.jsonl \
  --workspace outputs/ascend310bpro_full \
  --include-modality
```

This works for current single-model predictions, older routed predictions, and
the earlier main/expert/final cascade JSONL.

## Stage 6: TZB Result Packaging

Script:

```bash
bash deployment/ascend310bpro/run_tzb_submission.sh
```

It prepares the folder requested by the testing rules:

```text
tzb_metrics.json
tzb_metrics.csv
npu_fps_summary.json
agent_summary.json
README.md
```

With before/after `.pt` checkpoints and fixed test YAML files, it evaluates:

```text
base test set      -> before mAP, after mAP, KRR
increment test set -> after-model New-mAP
NPU summary        -> FPS
```

If the official `evaluate_tzb.py` is run separately, call it with `--skip-map`
to package only FPS and report evidence:

```bash
bash deployment/ascend310bpro/run_tzb_submission.sh \
  --skip-map \
  --npu-summary outputs/ascend310bpro_full/best_6class_infer/summary.json \
  --agent-summary outputs/ascend310bpro_full/agent_reports/agent_summary.json \
  --output-dir outputs/ascend310bpro_full/tzb_submission
```

## Competition Before/After Flow

Use this flow when the test rule asks for an increment-before model and an
increment-after model:

```text
PC:
  train_before_model_pc.ps1
    data/datasets_r1_base_train -> before 4-class best.pt + before ONNX

Board:
  run_increment_from_before.sh
    before 4-class best.pt + labeled six-class increment data -> after best.pt + after ONNX

Board:
  run_tzb_board_tests.sh
    before ONNX on data/testdata/base_test_r1
    after ONNX  on data/testdata/base_test_r1
    after ONNX  on data/testdata/inc_test_r2
```

`data/testdata` currently contains image folders only. It is suitable for NPU
prediction/FPS evidence, but it cannot be used by local PyTorch evaluation to
compute mAP/KRR/New-mAP unless labels or the official `evaluate_tzb.py` are
provided.
