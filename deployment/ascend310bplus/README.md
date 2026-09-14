# Ascend 310B Plus Six-Class Inference

This is a standalone board-side runtime project. It implements the routing
logic from `ROUTING_LOGIC_DETAILED.md` without modifying the older
`deployment/ascend310b` backup.

## Model Layout

For the current single six-class detector, put `best.onnx` here:

```text
deployment/ascend310bplus/models/best.onnx
```

If you keep it as `deployment/best.onnx`, the scripts will also find it
automatically.

Run it with:

```bash
bash deployment/ascend310bplus/run_best_6class_npu.sh \
  --model deployment/ascend310bplus/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bplus/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/best_6class
```

The class order is fixed by `classes_6.txt`:

```text
soldier
small_aircraft
warship
tank
patrol_boat
armored_vehicle
```

The older routed deployment still supports three exported ONNX or OM files:

```text
models/
  01_scene_router_224.onnx
  02_easy_detector_6class_640.onnx
  03_hard_detector_3class_960.onnx
```

The same filenames are also accepted under `deployment/ascend310bplus/models/`.
The default paths are configured as `models/...` in `config.json`; the runtime
first checks paths relative to this folder and then the project root. You can
also override model paths on the command line.

Check the current single six-class model before converting:

```bash
bash deployment/ascend310bplus/check_models.sh \
  --soc-version Ascend310B4
```

For the older routed deployment, add `--routed`:

```bash
bash deployment/ascend310bplus/check_models.sh \
  --routed \
  --scene-model models/your_scene_router.onnx \
  --easy-model models/your_easy_detector.onnx \
  --hard-model models/your_hard_detector.onnx \
  --soc-version Ascend310B4
```

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

## Full Pipeline From Raw Dataset

The full board wrapper starts from a YOLO dataset, builds the deterministic
offline augmentation, converts `best.onnx` to OM, then runs six-class
detection:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --soc-version Ascend310B4
```

`data/datasets_r1_base_train` can be a flat original dataset with images,
matching YOLO `.txt` labels, and `classes.txt` in the same folder. It does not
need to contain `data.yaml`; the wrapper generates one in the augmented output.
The wrapper uses `deployment/ascend310bplus/classes_6.txt` by default so the
generated dataset keeps the six-class protocol even if the source class file is
stale or incomplete.

The generated augmented dataset is:

```text
outputs/ascend310bplus_full/augmented_dataset/data.yaml
```

The inference outputs are:

```text
outputs/ascend310bplus_full/best_6class_infer/summary.json
outputs/ascend310bplus_full/best_6class_infer/predictions.jsonl
outputs/ascend310bplus_full/best_6class_infer/images/
```

If the augmented data already exists, reuse it:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --reuse-augmented \
  --soc-version Ascend310B4
```

To regenerate the augmented dataset in the same workspace, move the old output
aside and rebuild it at the same path:

```bash
bash deployment/ascend310bplus/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bplus_full \
  --rebuild-augmented \
  --soc-version Ascend310B4
```

If the filenames do not start with `ir_` or `sar_`, pass `--default-modality
ir` or `--default-modality sar`. More details are in `PIPELINE.md`.

To run only augmentation:

```bash
bash deployment/ascend310bplus/run_augment_yolo.sh \
  --data data/datasets_r1_base_train \
  --output outputs/ascend310bplus_full/augmented_dataset
```

## Optional Board-Side Class-IL Training

`run_class_il_training.sh` wraps the Windows PowerShell training settings in a
path-parameterized Bash script. The Ascend 310B path that is most reliable for
this project is still inference; full YOLO training on the board should be
treated as slow on CPU or experimental on `torch_npu`.

Run DER and ER with paths derived from one private data folder:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bplus/run_class_il_training.sh \
  --private-root "$PWD/数据集（不上传git）" \
  --method both
```

If the dataset location changes, pass the paths directly:

```bash
bash deployment/ascend310bplus/run_class_il_training.sh \
  --data "/path/to/yolo_r1_r2inc_augmented_full_tvt_seed42" \
  --prepared "/path/to/class_il_prepared_sparse_moe_seed42" \
  --initial-model "/path/to/yolo26n.pt" \
  --output-root "/path/to/runs" \
  --method der
```

The script defaults to `--device cpu`, `--batch-size 2`, and `--workers 0` for
the board. If a prepared directory already exists, reuse it explicitly:

```bash
bash deployment/ascend310bplus/run_class_il_training.sh \
  --private-root "$PWD/数据集（不上传git）" \
  --skip-prepare \
  --method er
```

For a quick environment smoke test:

```bash
bash deployment/ascend310bplus/run_class_il_training.sh \
  --private-root "$PWD/数据集（不上传git）" \
  --method der \
  --smoke-test
```

Final weights are written under:

```text
runs/class_il_sparse_moe_der_b500_e30_seed42/stage_06_armored_vehicle/weights/best.pt
runs/class_il_sparse_moe_er_b500_e30_seed42/stage_06_armored_vehicle/weights/best.pt
```

## Convert ONNX To OM

The inference script converts ONNX models automatically. To convert the
current single six-class `best.onnx` first:

```bash
bash deployment/ascend310bplus/convert_models.sh \
  --soc-version Ascend310B4
```

For the older routed deployment, add `--routed` and pass the three model paths:

```bash
bash deployment/ascend310bplus/convert_models.sh \
  --routed \
  --scene-model models/01_scene_router_224.onnx \
  --easy-model models/02_easy_detector_6class_640.onnx \
  --hard-model models/03_hard_detector_3class_960.onnx \
  --soc-version Ascend310B4
```

If the matching OM file already exists, it is reused. Pass `--force-convert` to
rebuild. The generated names include input size and SOC, for example:

```text
best_320x320_Ascend310B4.om
```

For `best.onnx`, the script auto-detects the static ONNX input size and passes
that size to ATC. Use `--width` and `--height` only when you deliberately want
to override it.

## Run Best Six-Class Inference

Run one image or a whole directory:

```bash
bash deployment/ascend310bplus/run_best_6class_npu.sh \
  --model deployment/ascend310bplus/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bplus/classes_6.txt \
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

## Optional Routed Inference

Use this only if you have the older scene/easy/hard model set:

```bash
bash deployment/ascend310bplus/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bplus_routed
```

Each routed image is assigned as follows:

```text
scene confidence < 0.60 -> hard detector
scene in air/sea       -> easy detector
scene in forest/urban  -> hard detector
```

Only one detector is executed per image.
During routed inference, the easy/hard detector is converted and loaded only
when at least one image is routed to that branch. Use
`convert_models.sh --routed` to convert all three models deliberately.

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
  "detector": {
    "route": "single",
    "route_reason": "single_6class_best_model",
    "elapsed_ms": 58.3,
    "input_size": [320, 320],
    "output_count": 1800
  },
  "detections": [
    {
      "box": [100.0, 120.0, 160.0, 190.0],
      "score": 0.81,
      "class_id": 0,
      "class_name": "soldier",
      "branch": "single"
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
