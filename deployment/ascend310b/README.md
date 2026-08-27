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
  --model detector_yolov8n_960_bs1.om \
  --image demo.png \
  --metadata package_metadata.json \
  --output result.json \
  --save-image outputs
```

For a directory of images, pass the directory to `--image`; JSON output will
contain one result object per image.

## Notes

- The runtime script is Python 3.9 compatible.
- The full project is still a training/export project and may require a newer
  Python plus PyTorch on the export machine.
- The OM model uses static batch=1 and NCHW float32 input.
- NMS runs on the CPU side in `infer_yolov8_om.py`.
