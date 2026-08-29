# Models

Place the three routed inference models in this directory:

```text
01_scene_router_224.onnx
02_easy_detector_6class_640.onnx
03_hard_detector_3class_960.onnx
```

The runtime can also use preconverted `.om` files. Update `../config.json` or
pass `--scene-model`, `--easy-model`, and `--hard-model` if your filenames are
different.
