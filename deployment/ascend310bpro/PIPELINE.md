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
bash deployment/ascend310bpro/run_model_manifest.sh verify
```

The default conversion path follows the report-aligned routed deployment:

```text
deployment/ascend310bpro/models/01_scene_router_224.onnx
deployment/ascend310bpro/models/02_easy_detector_6class_640.onnx
deployment/ascend310bpro/models/03_hard_detector_3class_960.onnx
```

The routed config is `route_config.yaml`: scene classification uses short-side
resize plus center crop to `224x224`; air/sea route to the easy six-class
detector at `640x640`; forest/urban and uncertain scenes route to the hard
three-class detector at `960x960`.

Matching OM files are cached by model name, size, and SOC, for example:

```text
02_easy_detector_6class_640_640x640_Ascend310B4.om
```

Existing OM files are reused. Pass `--force-convert` only when you really want
to rebuild them.

If ATC fails with `np.float_ was removed in the NumPy 2.0 release`, it is a
CANN/NumPy compatibility problem rather than a model-shape problem. If it fails
with `No module named 'attr'`, the CANN `te/opc-tool` Python dependencies are
incomplete. The scripts enable `deployment/ascend310bpro/atc_compat/sitecustomize.py`
automatically for ATC subprocesses; if the board environment is still missing
packages, run:

```bash
python3 -m pip install -r deployment/ascend310bpro/requirements-runtime.txt
```

After replacing any routed ONNX file, rebuild and verify the model manifest:

```bash
bash deployment/ascend310bpro/run_model_manifest.sh build --soc-version Ascend310B4
bash deployment/ascend310bpro/run_model_manifest.sh verify
```

The single six-class `best.onnx` baseline is still available:

```bash
bash deployment/ascend310bpro/convert_models.sh \
  --single \
  --model deployment/ascend310bpro/models/best.onnx \
  --soc-version Ascend310B4
```

INT8 calibration is not automated in this folder yet. The runnable deployment
path is ONNX to OM with the Ascend ATC/CANN runtime. Add INT8 only after AMCT
and a fixed calibration set are available.

## Stage 4: NPU Inference

Default routed inference:

```bash
bash deployment/ascend310bpro/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_routed
```

Outputs:

```text
summary.json
predictions.jsonl
images/
```

The single-model baseline is still available with `--single` or
`run_best_6class_npu.sh`:

```text
best.onnx -> six-class detection
```

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
  --predictions outputs/ascend310bpro_full/routed_infer/predictions.jsonl \
  --workspace outputs/ascend310bpro_full \
  --include-modality
```

This works for routed predictions, single-model baseline predictions, and
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
  --npu-summary outputs/ascend310bpro_full/routed_infer/summary.json \
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

## Stage 7: Test-Protocol Metrics

For the current release weights, use:

```bash
bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 作品编号 \
  --no-save-images
```

This runs the required before/base and after/base/increment predictions twice:
once with `class-il-er500-stage06-best.pt`, and once with
`class-il-der500-stage06-best.pt`. The formatted `目标检测识别模块` directories
contain normalized YOLO `class x_center y_center width height confidence` TXT
files and can be passed to the official `evaluate_tzb.py`.

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
