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

## Stage 6: Test-Protocol Metrics

After board inference, run `run_evaluate_tzb.sh` with a fixed base test set,
fixed incremental test set, and three prediction files: before-increment on
the base set, after-increment on the base set, and after-increment on the new
set.  The evaluator computes base mAP, old-class KRR, new-class New-mAP, and
reads FPS from the inference `summary.json`.  It writes both a machine-readable
JSON report and a four-row CSV scorecard.

```bash
bash deployment/ascend310bpro/run_evaluate_tzb.sh \
  --base-data data/base_test \
  --new-data data/incremental_test \
  --before-predictions outputs/before/predictions.jsonl \
  --after-base-predictions outputs/after_base/predictions.jsonl \
  --after-new-predictions outputs/after_new/predictions.jsonl \
  --fps-summary outputs/after_new/summary.json \
  --output outputs/ascend310bpro_tzb_metrics/metrics.json
```

The base set must be identical for the before/after runs. The evaluator uses
COCO-style 101-point AP at IoU 0.50 and the mean over IoU 0.50:0.95. KRR is
`old-mAP-after / old-mAP-before`; New-mAP is computed only over the new class
IDs. Missing labels, missing class support, or missing FPS evidence prevent
`evaluation_ready=true`.
