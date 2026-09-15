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

For the day-to-day command order, read `NORMAL_FLOW_USAGE.md`.

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

Default routed model set:

```text
deployment/ascend310bpro/models/01_scene_router_224.onnx
deployment/ascend310bpro/models/02_easy_detector_6class_640.onnx
deployment/ascend310bpro/models/03_hard_detector_3class_960.onnx
```

The routing protocol is defined by `deployment/ascend310bpro/route_config.yaml`.
It uses air/sea as the easy branch and forest/urban plus low-confidence scenes
as the hard branch. `best.onnx` is kept as an optional single-detector baseline.

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
```

ATC/CANN on some Ascend images is not compatible with NumPy 2.x because CANN
imports old NumPy aliases such as `np.float_`. The pro scripts automatically
add `deployment/ascend310bpro/atc_compat/sitecustomize.py` to the ATC
subprocess `PYTHONPATH` before ONNX-to-OM conversion. If ATC still reports
`np.float_ was removed in the NumPy 2.0 release` or `No module named 'attr'`,
use the environment fix:

```bash
python3 -m pip install -r deployment/ascend310bpro/requirements-runtime.txt
```

If you are running the routed three-ONNX package, also check:

```bash
bash deployment/ascend310bpro/check_models.sh --soc-version Ascend310B4
bash deployment/ascend310bpro/run_model_manifest.sh verify
```

After replacing any routed ONNX model, rebuild the SHA manifest:

```bash
bash deployment/ascend310bpro/run_model_manifest.sh build --soc-version Ascend310B4
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
outputs/ascend310bpro_full/routed_infer/summary.json
outputs/ascend310bpro_full/routed_infer/predictions.jsonl
outputs/ascend310bpro_full/routed_infer/images/
outputs/ascend310bpro_full/agent_reports/reports/*.json
outputs/ascend310bpro_full/agent_reports/batch_summary.csv
outputs/ascend310bpro_full/agent_reports/agent_summary.json
outputs/ascend310bpro_full/agent_memory.jsonl
```

## Only Run Inference

For a live single-image demo, use the release ER/DER checkpoint wrapper:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bpro/run_single_image_demo.sh \
  --image data/testdata/base_test_r1/example.jpg \
  --strategy er500 \
  --soc-version Ascend310B4
```

Use `--strategy der500` to demonstrate the DER increment model, or
`--strategy base` to demonstrate the before-increment four-class model. The
annotated image is written to:

```text
outputs/ascend310bpro_demo/<strategy>/images/<input-name>.jpg
```

If augmentation already exists:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --infer-only \
  --infer-input outputs/ascend310bpro_full/augmented_dataset \
  --workspace outputs/ascend310bpro_full \
  --soc-version Ascend310B4 \
  --include-modality
```

Or call routed inference directly:

```bash
bash deployment/ascend310bpro/run_routed_infer.sh \
  --input data/datasets_r1_base_train \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_routed
```

To run the optional single six-class baseline instead:

```bash
bash deployment/ascend310bpro/run_best_6class_npu.sh \
  --model deployment/ascend310bpro/models/best.onnx \
  --input data/datasets_r1_base_train \
  --classes deployment/ascend310bpro/classes_6.txt \
  --soc-version Ascend310B4 \
  --output-dir outputs/ascend310bpro_best
```

The single-model script auto-detects the static ONNX input size. For
`best.onnx`, this avoids the earlier ATC reshape error caused by forcing
`960x960` on a model exported as another size.

## Model Manifest

`model_manifest.json` records the three routed ONNX hashes, expected input
sizes, class lists, scene preprocessing, and hard-branch class remap. Rebuild it
after exporting new routed models:

```bash
bash deployment/ascend310bpro/run_model_manifest.sh build --soc-version Ascend310B4
bash deployment/ascend310bpro/run_model_manifest.sh verify
```

## INT8 Quantization Status

The current board package performs ONNX to OM conversion and NPU inference, but
does not claim INT8 calibration automatically. Use the default FP16/ATC
deployment for runnable evidence first. Add INT8 only after the Ascend AMCT
toolchain and a fixed calibration set are available, then rebuild the manifest
for the quantized artifacts.

## Only Build Agent Reports

This is useful when NPU inference already finished:

```bash
bash deployment/ascend310bpro/run_agent_reports.sh \
  --predictions outputs/ascend310bpro_full/routed_infer/predictions.jsonl \
  --summary outputs/ascend310bpro_full/routed_infer/summary.json \
  --output-dir outputs/ascend310bpro_full/agent_reports \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl \
  --rewrite-memory \
  --include-modality
```

It supports the routed output, the single-model baseline output, and the
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
the matching routed branch path in `deployment/ascend310bpro/models/`, or under
`deployment/ascend310bpro/models/best.onnx` for the optional single baseline,
then rerun conversion/inference.

## PC Before-Model Training

The competition protocol needs an increment-before model. Train it on the PC
from the labeled four-class base dataset with the `zzjDEREnv` conda env:

```powershell
Set-Location "D:\000zzjCodes\NCEPUWorkspace\TiaoZhanBeiWorkspace\ChallengeCup"

.\deployment\ascend310bpro\train_before_model_pc.ps1 `
  -CondaEnv zzjDEREnv `
  -DataRoot "data\datasets_r1_base_train" `
  -Workspace "outputs\ascend310bpro_before_pc" `
  -InitialModel "deployment\ascend310bpro\models\yolo26n.pt" `
  -Epochs 30 `
  -BatchSize 8 `
  -Workers 4 `
  -Device 0 `
  -ReuseAugmented
```

Outputs:

```text
outputs/ascend310bpro_before_pc/runs/base_before_r1/weights/best.pt
outputs/ascend310bpro_before_pc/exports/before_increment_4class_640x640.onnx
outputs/ascend310bpro_before_pc/before_model_paths.json
```

Copy the `.pt` checkpoint to the board for incremental training. Copy the ONNX
too if you want to run the before-model NPU blind test.

## Board Increment From Before

Run this on the Ascend 310B board after copying the four-class before `.pt`.
The increment data must be labeled and six-class; `data/testdata` is images
only and is not trainable.

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bpro/run_increment_from_before.sh \
  --base-data data/datasets_r1_base_train \
  --increment-data /path/to/six_class_increment_train_or_data.yaml \
  --before-model deployment/ascend310bpro/models/before_increment_4class.pt \
  --workspace outputs/ascend310bpro_increment \
  --method der \
  --buffer-size 500 \
  --device cpu \
  --batch-size 2 \
  --workers 0
```

The script prepares the four-to-six incremental protocol, trains the after
model, and exports:

```text
outputs/ascend310bpro_increment/runs/batch_il_der_b500/batch_incremental_training_summary.json
outputs/ascend310bpro_increment/runs/batch_il_der_b500/exports/after_increment_6class_640x640.onnx
```

For a first board check, add `--smoke-test`. If Sparse-MoE export or training
is unstable on the board, add `--no-sparse-moe`.

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

## TZB Test Result Folder

The requirement screenshot asks for:

```text
base test set: before-model and after-model mAP, then KRR
increment test set: after-model New-mAP
FPS: measured on Ascend 310B and submitted with evidence
```

For the current release weight list, run both increment strategies on the
Ascend board:

```bash
cd ~/Desktop/workspace/ChallengeCup
conda activate cc_env

bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
  --soc-version Ascend310B4 \
  --team-id 作品编号 \
  --no-save-images
```

It uses:

```text
before/base: deployment/ascend310bpro/models/r1-four-class-detector-easy-640.pt
after ER500 : deployment/ascend310bpro/models/class-il-er500-stage06-best.pt
after DER500: deployment/ascend310bpro/models/class-il-der500-stage06-best.pt
```

The script exports `.pt` checkpoints to cached ONNX files, converts/reuses OM
files, runs NPU inference, and writes the attachment-1 TXT folders:

```text
outputs/ascend310bpro_release_tzb/er500/作品编号_FPS指标<自动取整FPS>/
outputs/ascend310bpro_release_tzb/der500/作品编号_FPS指标<自动取整FPS>/
```

If you want to force the folder number, rerun with `--fps-label-er 30` and
`--fps-label-der 30`.

When you have the two `.pt` checkpoints and the fixed test YAML files:

```bash
bash deployment/ascend310bpro/run_tzb_submission.sh \
  --base-data /path/to/base_test_data.yaml \
  --increment-data /path/to/increment_test_data.yaml \
  --before-model /path/to/increment_before.pt \
  --after-model /path/to/increment_after.pt \
  --npu-summary outputs/ascend310bpro_full/routed_infer/summary.json \
  --agent-summary outputs/ascend310bpro_full/agent_reports/agent_summary.json \
  --output-dir outputs/ascend310bpro_full/tzb_submission \
  --device cpu \
  --image-size 640
```

If the official `evaluate_tzb.py` is provided separately and will compute the
mAP metrics, package only the board-side FPS and Agent evidence:

```bash
bash deployment/ascend310bpro/run_tzb_submission.sh \
  --skip-map \
  --npu-summary outputs/ascend310bpro_full/routed_infer/summary.json \
  --agent-summary outputs/ascend310bpro_full/agent_reports/agent_summary.json \
  --output-dir outputs/ascend310bpro_full/tzb_submission
```

The result folder contains:

```text
tzb_metrics.json
tzb_metrics.csv
npu_fps_summary.json
agent_summary.json
README.md
```

For the unlabeled image folders currently in this repo:

```bash
bash deployment/ascend310bpro/run_tzb_board_tests.sh \
  --before-model deployment/ascend310bpro/models/before_increment_4class_640x640.onnx \
  --after-model outputs/ascend310bpro_increment/runs/batch_il_der_b500/exports/after_increment_6class_640x640.onnx \
  --workspace outputs/ascend310bpro_tzb_board \
  --soc-version Ascend310B4 \
  --before-decode-mode raw \
  --after-decode-mode raw
```

This writes `base_before`, `base_after`, and `increment_after` prediction
folders for `data/testdata/base_test_r1` and `data/testdata/inc_test_r2`.

## Single-Model Baseline

The optional single six-class `best.onnx` deployment is still available:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_single \
  --single \
  --reuse-augmented \
  --soc-version Ascend310B4
```

Use it as a baseline or emergency fallback. The report-aligned default is the
scene/easy/hard routed deployment in `route_config.yaml`.

## Required Test Indicators

The board submission protocol is implemented by `evaluate_tzb.py` and uses
the same fixed test images and NPU prediction JSONL files that were actually
run on Ascend 310B:

```text
base test + before model -> base mAP
base test + after model  -> after-base mAP
base before/after mAP    -> KRR = after old-class mAP / before old-class mAP
new test + after model   -> New-mAP (new classes only)
inference summary.json   -> FPS
```

Run it after collecting the three prediction files:

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

The CSV scorecard contains exactly `mAP@0.5`, `KRR`, `New-mAP`, and `FPS`; the
JSON also preserves `mAP-before` and `mAP-after` for auditability.  `summary.json` now records the measured FPS and its scope.  The
evaluator refuses to mark the report ready when a class has no ground-truth
support or FPS evidence is missing; it never substitutes training accuracy for
the required test metrics.
