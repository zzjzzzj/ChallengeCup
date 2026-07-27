from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_recognition.target_classifier_module.training import (
    VALID_AUGMENTATIONS,
    TrainingConfig,
    train_target_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练ResNet18真实框目标裁剪四分类基线"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("image_processing/artifacts/target_crops/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scene_recognition/target_classifier_module/runs/resnet18_target_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augmentation", choices=VALID_AUGMENTATIONS, default="none")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        device=args.device,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    result = train_target_classifier(args.manifest, args.output, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
