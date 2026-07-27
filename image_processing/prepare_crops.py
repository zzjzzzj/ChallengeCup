from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_recognition.target_classifier_module.dataset import build_target_crop_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据YOLO真实标注框生成ResNet18四类目标裁剪数据"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("image_processing/artifacts/scene_index.csv"),
        help="包含image_path/split/sensor/scene字段的原图索引",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_processing/artifacts/target_crops"),
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.10,
        help="在真实框四周额外保留的上下文比例，默认10%%",
    )
    args = parser.parse_args()
    result = build_target_crop_dataset(args.index, args.output, args.padding_ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
