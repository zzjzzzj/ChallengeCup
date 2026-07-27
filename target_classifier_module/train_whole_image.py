from __future__ import annotations

import argparse
import json
from pathlib import Path

from target_classifier_module.whole_image import (
    WholeImageTrainingConfig,
    train_whole_image_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a whole-image ResNet18 multi-label target classifier")
    parser.add_argument("--manifest-dir", type=Path, required=True, help="目录内必须包含train.txt、val.txt、test.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation",
        choices=("none", "flip", "rotate90", "invert", "open", "close"),
        default="none",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    result = train_whole_image_classifier(
        args.manifest_dir,
        args.output,
        WholeImageTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            num_workers=args.num_workers,
            augmentation=args.augmentation,
            device=args.device,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
