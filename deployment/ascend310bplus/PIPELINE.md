# Ascend 310B Plus Full Pipeline

This folder is now the board-side project wrapper for:

```text
original YOLO dataset
  -> offline train-set augmentation
  -> optional board-side Class-IL training
  -> ONNX to OM conversion
  -> single six-class best.onnx inference
  -> optional scene recognition and route-based model matching
  -> detection outputs
```

The heavy model definitions and training code still live in the main project.
The board wrapper calls those stable entry points instead of copying them.

## What Is Implemented

### 1. Dataset Augmentation

Script:

```bash
bash deployment/ascend310bplus/run_augment_yolo.sh
```

Implementation:

```text
run_augment_yolo.sh
  -> python train.py augment-yolo
  -> scene_recognition.detector_module.augment_yolo_dataset
  -> deployment.ascend310b.augment_selected_yolo augmentation recipe
```

The augmentation is offline and deterministic. Only the train split is
augmented; val/test are copied without augmentation.

IR image operations:

```text
ir_gamma_bright
invert_255
rot180
```

SAR image operations:

```text
rot180
sar_rot90_cw
sar_gamma
```

For rotation operations, YOLO labels are transformed together with the image.
The output dataset contains:

```text
data.yaml
classes.txt
images/train, labels/train
images/val, labels/val
images/test, labels/test       # if the source has test
augmentation_manifest.csv
augmentation_summary.json
```

`run_augment_yolo.sh` uses `deployment/ascend310bplus/classes_6.txt` by default:

```text
soldier
small_aircraft
warship
tank
patrol_boat
armored_vehicle
```

This keeps the generated `data.yaml` on the six-class protocol even when an
older flat source directory contains a stale four-class `classes.txt`.

### 2. Optional Class-IL Training

Script:

```bash
bash deployment/ascend310bplus/run_class_il_training.sh
```

This wraps:

```text
python train.py prepare-class-il
python train.py class-il-yolo
```

The default board settings are conservative:

```text
device=cpu
batch-size=2
workers=0
```

Use `--train-device npu:0` only after verifying that `torch_npu` and
Ultralytics training work on the board. `--device 0` usually means CUDA device
0, not Ascend NPU.

### 3. Scene Recognition And Model Matching

The current `best.onnx` model is a single six-class detector. Use this path when
you have only `best.onnx`:

```bash
bash deployment/ascend310bplus/run_best_6class_npu.sh \
  --model deployment/ascend310bplus/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bplus/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/best_6class
```

This single-model path decodes compact detector outputs:

```text
x1, y1, x2, y2, confidence, class_id
```

If you also have the three routed models, use the routed script:

Script:

```bash
bash deployment/ascend310bplus/run_routed_infer.sh
```

Implementation:

```text
routed_infer_npu.py
  -> scene router model
  -> route decision
  -> easy or hard detector
```

Default model files:

```text
deployment/ascend310bplus/models/01_scene_router_224.onnx
deployment/ascend310bplus/models/02_easy_detector_6class_640.onnx
deployment/ascend310bplus/models/03_hard_detector_3class_960.onnx
```

Routing rule:

```text
scene confidence < 0.60 -> hard detector
air or sea               -> easy detector
forest or urban          -> hard detector
```

Only one detector runs for each image. The easy detector returns six-class
YOLOv10 end-to-end boxes. The hard detector decodes three-class YOLOv8 raw
outputs and maps local classes back to the global six-class IDs:

```text
soldier         -> soldier
tank            -> tank
armored_vehicle -> armored_vehicle
```

### 4. Complete Board Wrapper

Script:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh
```

Default stages:

```text
1. augment original data
2. skip board training unless --train is passed
3. convert ONNX to OM, reusing existing OM
4. run best.onnx single-model inference, or routed inference when --routed is passed
```

## Minimal Commands On The Board

Install runtime dependencies:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env
python3 -m pip install -r deployment/ascend310bplus/requirements-runtime.txt
```

Run the whole board-side flow:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --soc-version Ascend310B4
```

`data/datasets_r1_base_train` may be the original flat dataset layout:

```text
classes.txt
ir_r1_base_air_000001.png
ir_r1_base_air_000001.txt
...
```

If there is no `data.yaml`, the wrapper uses the flat-root augmentation helper
and writes a new YOLO `data.yaml` in the augmented dataset directory.

If the image filenames do not start with `ir_` or `sar_`, add one default
modality:

```bash
--default-modality ir
```

If the augmented dataset already exists:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --reuse-augmented \
  --soc-version Ascend310B4
```

If you want to regenerate in the same workspace:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --rebuild-augmented \
  --soc-version Ascend310B4
```

If you want inference on the original images instead of the augmented dataset:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --infer-input data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --soc-version Ascend310B4
```

## Optional Training Command

Board-side training is slow and depends on the training environment. To try it:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data "/path/to/original_yolo_dataset" \
  --workspace outputs/ascend310bplus_full \
  --train \
  --initial-model "/path/to/yolo26n.pt" \
  --training-method der \
  --train-device cpu \
  --train-batch-size 2 \
  --train-workers 0 \
  --soc-version Ascend310B4
```

The generated Class-IL checkpoint is not automatically used as `best.onnx` or
as the routed three-model deployment. Export the chosen checkpoint to ONNX
first, then pass that ONNX file as `--single-model`.

## What Still Needs External Preparation

The board-side project now has augmentation, OM conversion, single-model
six-class inference, scene recognition, and routed model matching.

For the corrected current path, the external model requirement is:

```text
best.onnx
```

Copy it to:

```text
deployment/ascend310bplus/models/best.onnx
```

If you want to use routed model matching instead of the single `best.onnx`
detector, the additional external requirement is the trained routed model set:

```text
01_scene_router_224.onnx
02_easy_detector_6class_640.onnx
03_hard_detector_3class_960.onnx
```

Their training recipes are documented under:

```text
scene_recognition/route_training/README.md
```

In normal use, train and export these ONNX files on a PC/GPU machine, copy them
to `deployment/ascend310bplus/models/`, and let the board convert/reuse OM
files for inference.
