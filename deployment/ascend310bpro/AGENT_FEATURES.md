# Agent Feature Comparison

This file records what was compared from `Agent/` and how the board-side
`ascend310bpro` package implements the useful parts.

## Added To `ascend310bpro`

| Agent capability | Source in `Agent/` | Pro implementation |
| --- | --- | --- |
| Batch report files | `Agent/cli.py batch` | `agent_outputs.py format` writes `agent_reports/reports/*.json`. |
| CSV run summary | `Agent/cli.py batch` and tests for result formatting | `agent_outputs.py` writes UTF-8-BOM `batch_summary.csv` for Excel/WPS. |
| Chinese output description | `Agent/agent.py` uses `build_image_summary` | Reimplemented in `build_image_summary`, with six-class Chinese names. |
| Feedback memory | `Agent/memory.py` and `Agent/cli.py feedback` | `run_feedback.sh` appends feedback to `agent_memory.jsonl`. |
| Memory summary | `EpisodeMemory.summary` | `agent_outputs.py memory-summary` and `agent_summary.json`. |
| Scene-target consistency | `Agent/reasoning.py` | Reimplemented with `ALLOWED_TARGETS_BY_SCENE` and final-scene fusion. |
| Runtime proxy loss | `Agent/losses.py` | Reimplemented as `runtime_proxy` loss in every report. |
| Model decision explanation | `Agent/reasoning.py build_decision` | `decision` field records route, confidence threshold, priority classes, and OM cache strategy. |
| Loss formula helper | `Agent/cli.py loss` | `agent_outputs.py loss` computes the same combined training-loss formula. |
| Training bridge | `Agent/cli.py` train bridges | Existing `run_class_il_training.sh` and `run_full_pipeline.sh --train` are kept and path-parameterized. |
| Competition result folder | requirement screenshot / `evaluate_tzb.py` mention | `run_tzb_submission.sh` packages mAP/KRR/New-mAP and NPU FPS evidence. |

## Kept From `ascend310bplus`

| Capability | Pro file |
| --- | --- |
| Report-aligned scene/easy/hard routed inference | `route_config.yaml`, `routed_infer_npu.py`, `run_routed_infer.sh` |
| ONNX to cached OM conversion | `routed_infer_npu.py`, `convert_models.sh` |
| Model SHA-256 manifest | `model_manifest.py`, `run_model_manifest.sh` |
| Optional single six-class baseline | `infer_best_6class_npu.py`, `run_best_6class_npu.sh` |
| Offline YOLO dataset augmentation | `run_augment_yolo.sh` |
| Board-side Class-IL wrapper | `run_class_il_training.sh` |

## Not Directly Moved

The following desktop Agent modules are not imported by pro:

```text
Agent/agent.py
Agent/image_ops.py
Agent/detection.py
Agent/scene.py
Agent/target.py
Agent/models/*
Agent/continual/*
```

Reason: those modules can pull in desktop training dependencies such as torch,
ultralytics, scene-recognition utilities, or checkpoint-specific code. The pro
deployment should stay stable on the Ascend board after NPU inference, so the
report layer is a lightweight reimplementation based only on prediction JSON.

## Output Contract

After `run_full_pipeline.sh`, the pro package produces:

```text
<workspace>/augmented_dataset/data.yaml
<workspace>/routed_infer/summary.json
<workspace>/routed_infer/predictions.jsonl
<workspace>/routed_infer/images/
<workspace>/agent_reports/reports/*.json
<workspace>/agent_reports/batch_summary.csv
<workspace>/agent_reports/agent_summary.json
<workspace>/agent_memory.jsonl
<workspace>/tzb_submission/tzb_metrics.json
<workspace>/tzb_submission/tzb_metrics.csv
<workspace>/tzb_submission/npu_fps_summary.json
```

Each per-image report contains:

```text
image
modality
scene
final_scene
detections
output_summary
consistency
decision
losses
memory
stages
source_prediction
```

## Typical Commands

Full run:

```bash
bash deployment/ascend310bpro/run_full_pipeline.sh \
  --data data/datasets_r1_base_train \
  --workspace outputs/ascend310bpro_full \
  --reuse-augmented \
  --soc-version Ascend310B4 \
  --include-modality
```

Report an already finished inference:

```bash
bash deployment/ascend310bpro/run_agent_reports.sh \
  --predictions outputs/ascend310bpro_full/routed_infer/predictions.jsonl \
  --summary outputs/ascend310bpro_full/routed_infer/summary.json \
  --output-dir outputs/ascend310bpro_full/agent_reports \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl \
  --rewrite-memory \
  --include-modality
```

Append feedback:

```bash
bash deployment/ascend310bpro/run_feedback.sh \
  --memory outputs/ascend310bpro_full/agent_memory.jsonl \
  --image data/datasets_r1_base_train/ir_r1_base_air_000001.png \
  --scene air \
  --modality ir \
  --targets small_aircraft \
  --note "manual correction"
```
