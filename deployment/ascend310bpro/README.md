# Ascend 310B Pro Deployment

`ascend310bpro` is the most complete board-side package in this repository. It
keeps the verified `ascend310bplus` inference/training wrappers, then adds a
lightweight Agent output layer for reports, CSV summaries, feedback memory, and
scene-target reasoning.

The older folders are left untouched:

```text
deployment/ascend310b      # backup / earlier cascade version
deployment/ascend310bplus  # stable plus version
deployment/ascend310bpro   # complete board-side version
```

## What It Runs

Default pro flow:

```text
raw YOLO dataset
  -> offline augmentation
  -> optional Class-IL incremental training
  -> ONNX to cached OM conversion
  -> Ascend NPU inference
  -> Agent-style reports / CSV / memory
```

Current default model path:

```text
deployment/ascend310bpro/models/best.onnx
```

If it is missing, scripts also try `deployment/best.onnx` and
`models/best.onnx`.

Class order is fixed by:

```text
deployment/ascend310bpro/classes_6.txt
```

```text
soldier
small_aircraft
warship
tank
patrol_boat
armored_vehicle
```

## Environment

On the Ascend board:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r deployment/ascend310bpro/requirements-runtime.txt
```

Check the runtime:

```bash
bash deployment/ascend310bpro/check_env.sh
bash deployment/ascend310bpro/check_models.sh --soc-version Ascend310B4
```

## One Command Full Pipeline

For your current board dataset:

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

Remove `--reuse-augmented` the first time if the augmented dataset does not
exist. Use `--rebuild-augmented` when you deliberately want to regenerate it.

Main outputs:

```text
outputs/ascend310bpro_full/augmented_dataset/data.yaml
outputs/ascend310bpro_full/best_6class_infer/summary.json
outputs/ascend310bpro_full/best_6class_infer/predictions.jsonl
outputs/ascend310bpro_full/best_6class_infer/images/
outputs/ascend310bpro_full/agent_reports/reports/*.json
outputs/ascend310bpro_full/agent_reports/batch_summary.csv
outputs/ascend310bpro_full/agent_reports/agent_summary.json
outputs/ascend310bpro_full/agent_memory.jsonl
```

## Only Run Inference

If augmentation already exists:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --infer-only \
  --infer-input outputs/ascend310bpro_full/augmented_dataset \
  --workspace outputs/ascend310bpro_full \
  --soc-version Ascend310B4 \
  --include-modality
```

Or call the detector directly:

```bash
bash deployment/ascend310bpro/run_best_6class_npu.sh \
  --model deployment/ascend310bpro/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bpro/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_best
```

The script auto-detects the static ONNX input size. For the current
`best.onnx`, this avoids the earlier ATC reshape error caused by forcing
`960x960` on a model exported as another size.

## Only Build Agent Reports

This is useful when NPU inference already finished:

```bash
bash deployment/ascend310bpro/run_agent_reports.sh \
  --predictions outputs/ascend310bpro_full/best_6class_infer/predictions.jsonl \
  --summary outputs/ascend310bpro_full/best_6class_infer/summary.json \
  --output-dir outputs/ascend310bpro_full/agent_reports \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl \
  --rewrite-memory \
  --include-modality
```

It supports the current single-model output, the older routed output, and the
earlier main/expert/final cascade output.

## Optional Incremental Training

Training starts from a trainable `.pt` or `.yaml`, not from `.onnx` or `.om`.
`yolo26n.pt` is already accepted from:

```text
deployment/ascend310bpro/models/yolo26n.pt
```

Quick smoke test:

```bash
bash deployment/ascend310bpro/run_class_il_training.sh \
  --data outputs/ascend310bpro_full/augmented_dataset/data.yaml \
  --prepared outputs/ascend310bpro_full/class_il_prepared \
  --initial-model deployment/ascend310bpro/models/yolo26n.pt \
  --output-root outputs/ascend310bpro_full/runs \
  --method der \
  --device cpu \
  --batch-size 2 \
  --workers 0 \
  --smoke-test
```

Full DER/ER training:

```bash
bash deployment/ascend310bpro/run_class_il_training.sh \
  --data outputs/ascend310bpro_full/augmented_dataset/data.yaml \
  --prepared outputs/ascend310bpro_full/class_il_prepared \
  --initial-model deployment/ascend310bpro/models/yolo26n.pt \
  --output-root outputs/ascend310bpro_full/runs \
  --method both \
  --device cpu \
  --batch-size 2 \
  --workers 0
```

After training, export the final `best.pt` to ONNX on the machine where your
Ultralytics export stack works, then place the new ONNX under
`deployment/ascend310bpro/models/best.onnx` and rerun conversion/inference.

## Feedback Memory

Append a manual correction:

```bash
bash deployment/ascend310bpro/run_feedback.sh \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl \
  --image data/datasets_r1_base_train/ir_r1_base_air_000001.png \
  --scene air \
  --modality ir \
  --targets small_aircraft \
  --note "manual correction"
```

Summarize memory:

```bash
python3 deployment/ascend310bpro/agent_outputs.py memory-summary \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl
```

## Routed Mode

The older scene/easy/hard routed deployment is still available:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_routed \
  --routed \
  --reuse-augmented \
  --soc-version Ascend310B4
```

Use it only when the three routed models are the desired strategy. The current
default is the single six-class `best.onnx`.
