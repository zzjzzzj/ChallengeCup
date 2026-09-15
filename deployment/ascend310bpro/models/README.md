# Models

For the report-aligned routed deployment, place these three ONNX files here:

```text
01_scene_router_224.onnx
02_easy_detector_6class_640.onnx
03_hard_detector_3class_960.onnx
```

They are described by `../route_config.yaml`. `run_full_pipeline.sh`,
`convert_models.sh`, and `check_models.sh` use this routed set by default.

For optional Class-IL incremental training, keep a trainable initial model here
as well:

```text
yolo26n.pt
```

Use `.pt` or `.yaml` for training. Use `.onnx` and `.om` only for export and
NPU inference.

The current TZB release-weight test entry expects these files:

```text
r1-four-class-detector-easy-640.pt
class-il-er500-stage06-best.pt
class-il-der500-stage06-best.pt
```

`run_release_weight_tzb_tests.sh` exports them to ONNX automatically on the
board and then runs the required before/after tests.

You may also keep the optional single six-class baseline in this directory:

```text
best.onnx
```

It is run with `deployment/ascend310bpro/run_best_6class_npu.sh` or with
`run_full_pipeline.sh --single --single-model deployment/ascend310bpro/models/best.onnx`.
The scripts also auto-detect `deployment/best.onnx` if that is where the
exported file is kept.

The runtime can also use preconverted `.om` files. Update `../route_config.yaml` or
pass `--scene-model`, `--easy-model`, and `--hard-model` if your filenames are
different.

The default `../route_config.yaml` also accepts the same names under the
project-root `models/` directory, which is often more convenient on the board:

```text
ChallengeCup/models/01_scene_router_224.onnx
ChallengeCup/models/02_easy_detector_6class_640.onnx
ChallengeCup/models/03_hard_detector_3class_960.onnx
```
