from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector_module.dataset import prepare_detection_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成目标检测数据清单与 YOLO 数据配置")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "scene_module" / "artifacts" / "scene_index.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "detector_module" / "artifacts" / "detection_dataset",
    )
    args = parser.parse_args()

    stats = prepare_detection_dataset(args.index, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
