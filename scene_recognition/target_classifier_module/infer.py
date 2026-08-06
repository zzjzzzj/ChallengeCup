from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from scene_recognition.target_classifier_module.training import (
    build_resnet18,
    build_transforms,
    resolve_device,
)


def load_crop_classifier(checkpoint_path: Path, device_name: str = "auto") -> dict[str, Any]:
    """Load a ResNet18 crop classifier checkpoint into a reusable runtime dict."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"目标分类检查点不存在: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") not in {None, "resnet18"}:
        raise ValueError(f"不支持的检查点模型: {checkpoint.get('model_name')}")
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", 224))
    device = resolve_device(device_name)
    model = build_resnet18(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    _, evaluation_transform = build_transforms(image_size, augmentation="none")
    return {
        "torch": torch,
        "model": model,
        "device": device,
        "transform": evaluation_transform,
        "class_names": class_names,
        "image_size": image_size,
        "checkpoint": str(checkpoint_path.resolve()),
    }


def predict_crop_image(runtime: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Classify one PIL crop using a loaded runtime."""

    torch_module = runtime["torch"]
    tensor = runtime["transform"](image.convert("RGB")).unsqueeze(0).to(runtime["device"])
    with torch_module.inference_mode():
        probabilities = runtime["model"](tensor).softmax(dim=1)[0].detach().cpu().tolist()
    class_names = list(runtime["class_names"])
    predicted_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    return {
        "predicted_id": predicted_id,
        "predicted": class_names[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            name: float(probability) for name, probability in zip(class_names, probabilities)
        },
    }


def predict_target_crop(
    image_path: Path,
    checkpoint_path: Path,
    device_name: str = "auto",
) -> dict:
    """Classify one already-cropped target image."""

    if not image_path.is_file():
        raise FileNotFoundError(f"目标裁剪图不存在: {image_path}")
    runtime = load_crop_classifier(checkpoint_path, device_name)
    with Image.open(image_path) as opened:
        prediction = predict_crop_image(runtime, opened)
    return {
        "image": str(image_path.resolve()),
        "predicted": prediction["predicted"],
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "scope": "输入必须是已裁剪目标；本命令不负责在完整图片中寻找目标框。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用ResNet18识别一张已裁剪目标图")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict_target_crop(args.image, args.checkpoint, args.device)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
