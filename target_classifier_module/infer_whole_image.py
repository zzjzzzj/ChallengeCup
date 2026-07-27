from __future__ import annotations

import argparse
import json
from pathlib import Path

from target_classifier_module.whole_image import predict_whole_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer target presence from one full image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = predict_whole_image(args.image, args.checkpoint, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
