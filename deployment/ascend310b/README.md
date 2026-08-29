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
