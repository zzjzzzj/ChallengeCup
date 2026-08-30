# Ascend 310B deployment

This directory builds a small runtime package for an aarch64 Ascend 310B device.
The device package runs an OM model with CANN ACL Python and does not need
PyTorch, torchvision, or Ultralytics.

## Build on the training/export machine

From the project root, use a Python environment that can import torch,
ultralytics, onnx, and PyYAML.

```bash
python train.py ascend310b-package \
  --checkpoint scene_recognition/detector_module/runs/ascend_yolov8n_960_aug_pretrained/weights/best.pt \
  --data scene_recognition/detector_module/artifacts/comparison_dataset/data_aug.yaml \
  --image-size 960 \
  --output dist/ascend310b_yolov8n_960 \
  --archive \
  --force
```

If you already have a static batch=1 ONNX file, package it directly:

```bash
python train.py ascend310b-package \
  --onnx path/to/detector_yolov8n_bs1.onnx \
  --classes data/datasets_r1_base_train/classes.txt \
  --image-size 960 \
  --output dist/ascend310b_yolov8n_960 \
  --archive \
  --force
```

Copy the output folder or zip archive to the Ascend device.

## Optional: augment before board-side training

This step is for a full project checkout on the microcomputer, not for the
runtime-only inference package. It uses the project dataset root instead of
hard-coded Windows paths.

Install the lightweight augmentation dependencies first. Install
torch/torchvision separately for your aarch64 environment before starting YOLO
training.

```bash
python3 -m pip install -r deployment/ascend310b/requirements-training.txt
```

Generate an augmented YOLO training dataset:

```bash
python3 train.py ascend310b-augment \
  --dataset-root data/datasets_r1_base_train \
  --output outputs/datasets_r1_base_train_augmented \
  --include-original \
  --classes data/datasets_r1_base_train/classes.txt
```

`--dataset-root` supports both standard YOLO folders
`images[/train] + labels[/train]` and this project's flat board-side folder
where images, same-name `.txt` labels, and `classes.txt` live together.

The output contains `images/`, `labels/`, `classes.txt`,
`augmentation_manifest.csv`, `augmentation_summary.json`, and `data.yaml`.
If no independent validation set is provided, the generated `data.yaml` points
`val` to the augmented training images so the training command can run. That is
useful for smoke testing, but it is not an independent accuracy metric.

If you have a separate validation dataset, pass it while generating the YAML:

```bash
python3 train.py ascend310b-augment \
  --dataset-root data/datasets_r1_base_train \
  --val-root data/datasets_r1_base_val \
  --output outputs/datasets_r1_base_train_augmented \
  --include-original \
  --classes data/datasets_r1_base_train/classes.txt
```

Then train with the generated YAML:

```bash
python3 train.py yolo \
  --data outputs/datasets_r1_base_train_augmented/data.yaml \
  --image-size 960 \
  --batch-size 4 \
  --workers 2 \
  --device cpu \
  --name ascend310b_augmented_yolov8n_960
```

Or run augmentation plus training in one command. If the augmented `data.yaml`
already exists, it is reused unless `--force-augment` is passed.

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

## Required microcomputer-side incremental training

Ascend 310B boards are the supported path for OM/ACL inference.  If the project
requires incremental training to happen on the microcomputer, use CPU PyTorch
for weight updates and keep the NPU for conversion/inference checks.

First check which Python is actually running and whether training dependencies
are visible:

```bash
bash deployment/ascend310b/run_probe_training_env.sh
```

If `torch` or `ultralytics` is missing, install them into the same environment
that will launch training.  Install torch/torchvision from a CPU/aarch64 wheel
source appropriate for your board, then install the project training extras:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deployment/ascend310b/requirements-training.txt
```

For board-side incremental fine-tuning, start with a smoke test, not a full
960-pixel run:

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

After the smoke test succeeds, increase only one budget at a time: first
`--epochs`, then `--image-size`, then remove or reduce `--freeze`. A practical
board-side run is usually `--image-size 640 --batch-size 1 --workers 0`.

For the six-stage Class-IL runner, use ER first because DER runs an extra
teacher model during training:

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

`--device npu:0` is accepted for experiments only when `torch_npu` is installed,
but it is not the recommended delivery path on Ascend 310B. Treat CPU training
plus NPU OM inference as the reliable microcomputer pipeline.

## Run ONNX on CPU first

Since ONNX CPU is often the easiest board-side smoke test, run this before OM:

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

This verifies the full project-level preprocessing, postprocessing, JSON, and
visualization flow on the microcomputer.

## Convert on the Ascend 310B device

```bash
cd ascend310b_yolov8n_960
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r requirements-runtime.txt

npu-smi info
atc --list_soc_version
export SOC_VERSION=Ascend310B4
bash convert_onnx_to_om.sh detector_yolov8n_bs1.onnx detector_yolov8n_960_bs1
```

Replace `Ascend310B4` with the exact value supported by the board.

## Run inference

```bash
bash run_infer.sh \
  --model detector_yolov8n_bs1.onnx \
  --image demo.png \
  --soc-version Ascend310B4 \
  --metadata package_metadata.json \
  --output result.json \
  --save-image outputs
```

For a directory of images, pass the directory to `--image`; JSON output will
contain one result object per image.
If `--model` is an ONNX file, the script converts it to OM with ATC first and
reuses the cached OM on later runs. You can still pass an existing `.om` file.

## Run the two-model cascade

Use this after both OM models are available on the board:

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

The script converts ONNX to OM first when needed. The default strategy then
runs the 1024x832 expert on every image, fuses the expert class with the
six-class result, writes `summary.json`, writes `predictions.jsonl`, and saves
annotated images under `cascade_outputs/images`.

If speed matters more than recall, trigger the expert only when the six-class
model misses the expert class or gives it low confidence:

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

If your OM output is the compact `300x6` postprocessed format like
`x1,y1,x2,y2,conf,class_id`, keep the defaults. For raw YOLO head output, pass
`--main-output-mode raw --expert-output-mode raw`.
If only the expert model uses a different compact format, keep the main model
unchanged and set only `--expert-nms-format`, for example
`--expert-nms-format xywh-conf-class`.

You can also pass ONNX models directly. The script will call `atc`, cache the
converted OM files, and then continue NPU inference:

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

The generated OM file names include input size and soc version, for example
`detector_6class_960_960x960_Ascend310B4.om`. If that OM file already exists,
it is reused. Pass `--force-convert` to rebuild it.

ATC uses NCHW input shape. If width/height are not passed, the script first
tries to read the ONNX input shape, then falls back to model names such as
`soldier_legacy4_1120x896.onnx`. For that model, the ATC shape becomes
`images:1,3,896,1120`.

You can still override sizes explicitly:

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

For a true single-class expert, `--expert-classes` is not needed. For a legacy
multi-class expert such as `soldier_legacy4_1120x896.onnx`, pass its class-name
file so the cascade keeps only the requested `--expert-class` instead of
treating every expert detection as soldier.

## Notes

- The runtime script is Python 3.9 compatible.
- The full project is still a training/export project and may require a newer
  Python plus PyTorch on the export machine.
- The OM model uses static batch=1 and NCHW float32 input.
- NMS runs on the CPU side in `infer_yolov8_om.py`.
