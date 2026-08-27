from __future__ import annotations

from pathlib import Path


YOLOV8N_P2_YAML = """# YOLOv8n-P2 for tiny-target continual detection
nc: 4
depth_multiple: 0.33
width_multiple: 0.25

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C2f, [128]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]
  - [[18, 21, 24, 27], 1, Detect, [nc]]
"""


def write_yolov8n_p2_yaml(path: Path, class_count: int = 4) -> Path:
    """Write an Ultralytics model YAML with P2/P3/P4/P5 detection outputs."""

    text = YOLOV8N_P2_YAML.replace("nc: 4", f"nc: {class_count}", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
