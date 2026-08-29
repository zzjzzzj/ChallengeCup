# Ascend 310B Plus Routed Inference

This is a standalone board-side runtime project. It implements the routing
logic from `ROUTING_LOGIC_DETAILED.md` without modifying the older
`deployment/ascend310b` backup.

## Model Layout

Put the three exported ONNX or OM files here:

```text
deployment/ascend310bplus/
  models/
    01_scene_router_224.onnx
    02_easy_detector_6class_640.onnx
    03_hard_detector_3class_960.onnx
```

The default paths are configured in `config.json`. You can edit that file or
override model paths on the command line.

## Runtime Dependencies

On the Ascend device:

```bash
cd ~/Desktop/workspace/ChallengeCup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r deployment/ascend310bplus/requirements-runtime.txt
```

Check the environment:

```bash
bash deployment/ascend310bplus/check_env.sh
```

## Convert ONNX To OM

The inference script converts ONNX models automatically. To convert first:

```bash
bash deployment/ascend310bplus/convert_models.sh \
  --soc-version Ascend310B4
```

If the matching OM file already exists, it is reused. Pass `--force-convert` to
rebuild. The generated names include input size and SOC, for example:

```text
01_scene_router_224_224x224_Ascend310B4.om
02_easy_detector_6class_640_640x640_Ascend310B4.om
03_hard_detector_3class_960_960x960_Ascend310B4.om
```

## Run Routed Inference

Run one image or a whole directory:

```bash
bash deployment/ascend310bplus/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bplus
```

If `--input` is a standard YOLO dataset root that contains an `images/`
subdirectory, only images under that subdirectory are scanned. This avoids
accidentally processing unrelated preview or output images stored next to the
dataset.

Outputs:

```text
outputs/ascend310bplus/
  summary.json
  predictions.jsonl
  images/
```

Each image is routed as follows:

```text
scene confidence < 0.60 -> hard detector
scene in air/sea       -> easy detector
scene in forest/urban  -> hard detector
```

Only one detector is executed per image.
During normal inference, the easy/hard detector is converted and loaded only
when at least one image is routed to that branch. `convert_models.sh` still
converts all three models deliberately.

## Useful Overrides

Use existing OM files directly:

```bash
bash deployment/ascend310bplus/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --scene-model deployment/ascend310bplus/models/01_scene_router_224.om \
  --easy-model deployment/ascend310bplus/models/02_easy_detector_6class_640.om \
  --hard-model deployment/ascend310bplus/models/03_hard_detector_3class_960.om \
  --output-dir outputs/ascend310bplus
```

Adjust thresholds:

```bash
bash deployment/ascend310bplus/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --route-confidence 0.60 \
  --easy-conf 0.25 \
  --hard-conf 0.25 \
  --hard-iou 0.55 \
  --output-dir outputs/ascend310bplus
```

If the hard model scores look like logits instead of probabilities:

```bash
--hard-score-activation sigmoid
```

If you only want JSON outputs:

```bash
--no-save-images
```

## Output Format

`predictions.jsonl` contains one JSON object per image:

```json
{
  "image": "...",
  "image_size": [640, 512],
  "scene": {
    "id": 1,
    "name": "forest",
    "confidence": 0.92,
    "scores": [0.01, 0.92, 0.02, 0.05],
    "elapsed_ms": 2.1
  },
  "detector": {
    "route": "hard",
    "route_reason": "confident_hard_scene",
    "elapsed_ms": 58.3,
    "input_size": [960, 960],
    "output_count": 132300
  },
  "detections": [
    {
      "box": [100.0, 120.0, 160.0, 190.0],
      "score": 0.81,
      "class_id": 0,
      "class_name": "soldier",
      "branch": "hard"
    }
  ]
}
```

Hard detector local classes are remapped to the global six-class IDs:

```text
0 soldier         -> 0 soldier
1 tank            -> 3 tank
2 armored_vehicle -> 5 armored_vehicle
```

## Notes

- The scene router uses RGB, NCHW, float32, normalized to `[0, 1]`.
- The easy detector uses letterbox to 640x640 and decodes `[x1,y1,x2,y2,conf,class_id]`.
- The hard detector uses letterbox to 960x960, decodes raw YOLOv8 `xywh + class scores`, then applies class-wise NMS.
- The route confidence threshold is separate from detector confidence.
