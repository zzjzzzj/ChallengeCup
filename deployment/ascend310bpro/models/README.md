# Models

For the corrected single six-class detector flow, place the exported detector
here:

```text
best.onnx
```

It is run with `deployment/ascend310bpro/run_best_6class_npu.sh` or with
`run_full_pipeline.sh --single-model deployment/ascend310bpro/models/best.onnx`.
The scripts also auto-detect `deployment/best.onnx` if that is where the
exported file is kept.

For optional Class-IL incremental training, keep a trainable initial model here
as well:

```text
yolo26n.pt
```

Use `.pt` or `.yaml` for training. Use `.onnx` and `.om` only for export and
NPU inference.

You may also place the three routed inference models in this directory:

```text
01_scene_router_224.onnx
02_easy_detector_6class_640.onnx
03_hard_detector_3class_960.onnx
```

The runtime can also use preconverted `.om` files. Update `../config.json` or
pass `--scene-model`, `--easy-model`, and `--hard-model` if your filenames are
different.

The default `../config.json` also accepts the same names under the project-root
`models/` directory, which is often more convenient on the board:

```text
ChallengeCup/models/01_scene_router_224.onnx
ChallengeCup/models/02_easy_detector_6class_640.onnx
ChallengeCup/models/03_hard_detector_3class_960.onnx
```
