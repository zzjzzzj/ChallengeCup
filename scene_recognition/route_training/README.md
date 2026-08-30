# Routed detector training code

This directory contains only the training entry points for the three ONNX
models in the current scene-routing deployment package. Datasets and model
weights stay outside Git and are supplied with command-line arguments or
environment variables.

## Models and recipes

| Branch | Script | Recipe | Deployed input |
|---|---|---|---:|
| Scene router | `train_scene_router_incremental.py` | YOLOv8n-cls, r1+r2 incremental fine-tuning, 224 | 224x224 |
| Easy detector | `train_easy_6class_yolov10_incremental.py` | YOLOv10n, six-class incremental fine-tuning at 960; export the checkpoint at 640 | 640x640 |
| Hard detector | `train_hard_3class_yolov8n_corrected.py` | COCO YOLOv8n, three classes, 960 square, Mosaic and shuffle enabled | 960x960 |

The three deployed ONNX files are intentionally not committed. They are
generated artifacts and are ignored by the repository's `.gitignore`.

## Commands

Run from the repository root after installing `ultralytics` and the matching
PyTorch/CUDA build:

```powershell
python scene_recognition/route_training/train_scene_router_incremental.py `
  --model D:/models/scene_cls_yolov8n_v2_best.pt `
  --data D:/scene_cls_r1_r2_incremental `
  --device 0

python scene_recognition/route_training/train_easy_6class_yolov10_incremental.py `
  --model D:/models/init6.pt `
  --data D:/yolo_r1_r2inc_augmented_full/yolo_r1_r2inc_augmented_full/data.yaml `
  --device 0

python scene_recognition/route_training/train_hard_3class_yolov8n_corrected.py `
  --model ./yolov8n.pt `
  --data D:/hard_3class_dataset/hard_3class.yaml `
  --device 0
```

The easy detector is trained at 960 because that is the checkpoint training
protocol. After selecting `best.pt`, export the deployed low-cost branch with
Ultralytics at 640:

```powershell
yolo export model=scene_recognition/runs/route_training/easy_6class_yolov10n_incremental_960/weights/best.pt format=onnx imgsz=640 batch=1 opset=12
```

The scene classifier's route-safe checkpoint must be selected against the
detector validation set using the route evaluation code; the classifier's
ordinary Top-1 checkpoint is not automatically the safest route checkpoint.

## Dataset expectations

- Scene router: Ultralytics classification layout with `train/` and `val/`,
  classes `air`, `forest`, `sea`, `urban`.
- Easy detector: six-class YAML in the order
  `soldier, small_aircraft, warship, tank, patrol_boat, armored_vehicle`.
- Hard detector: three-class YAML in the order
  `soldier, tank, armored_vehicle`.

Use the original 183-image validation split for final comparison. Do not copy
augmented images into that validation split.
