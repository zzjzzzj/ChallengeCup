# TZB Test Result Folder

This folder follows attachment 1 of the test/submission requirements.

Required prediction groups:

- `基础模型-基础测试集推理结果/目标检测识别模块`: 240 image TXT files
- `增量模型-基础测试集推理结果/目标检测识别模块`: 240 image TXT files
- `增量模型-增量测试集推理结果/目标检测识别模块`: 270 image TXT files

Detection TXT format:

```text
class x_center y_center width height confidence
```

Coordinates are normalized YOLO xywh values.
