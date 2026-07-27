from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from train_scene_classifier import SCENES, build_model


DEFAULT_POLICY = {
    "air": {
        "detector_profile": "air_small_target",
        "priority_classes": ["small_aircraft"],
        "confidence_threshold": 0.30,
        "notes": "保留弱小目标细节，避免过度平滑",
    },
    "sea": {
        "detector_profile": "sea_ship",
        "priority_classes": ["warship"],
        "confidence_threshold": 0.35,
        "notes": "关注海面孤立结构；SAR高噪声时优先抑制散斑",
    },
    "urban": {
        "detector_profile": "urban_occlusion",
        "priority_classes": ["soldier", "tank"],
        "confidence_threshold": 0.40,
        "notes": "避免强边缘增强造成建筑纹理误检",
    },
    "forest": {
        "detector_profile": "forest_complex_background",
        "priority_classes": ["soldier", "tank"],
        "confidence_threshold": 0.30,
        "notes": "抑制复杂纹理干扰并保留局部热特征",
    },
    "uncertain": {
        "detector_profile": "general",
        "priority_classes": ["soldier", "small_aircraft", "warship", "tank"],
        "confidence_threshold": 0.35,
        "notes": "场景置信度不足，使用通用保守策略",
    },
}


def quality_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as im:
        gray_im = im.convert("L")
        gray = np.asarray(gray_im, dtype=np.float32)
        blurred = np.asarray(gray_im.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    residual = gray - blurred
    q01, q99 = np.percentile(gray, [1, 99])
    return {
        "mean_gray": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "dynamic_range_p01_p99": float(q99 - q01),
        "sharpness_gradient": float((np.abs(gx).mean() + np.abs(gy).mean()) / 2.0),
        "high_frequency_noise": float(np.mean(np.abs(residual))),
    }


def level(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    return "high"


def calibrate(train_csv: Path, output: Path) -> None:
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8-sig", newline="")))
    metrics_by_sensor: dict[str, dict[str, list[float]]] = {}
    for idx, row in enumerate(rows, start=1):
        metrics = quality_metrics(Path(row["image_path"]))
        sensor = row["sensor"]
        metrics_by_sensor.setdefault(sensor, {k: [] for k in metrics})
        for key, value in metrics.items():
            metrics_by_sensor[sensor][key].append(value)
        if idx % 100 == 0:
            print(f"calibrated {idx}/{len(rows)}")
    result = {"source": str(train_csv.resolve()), "quantiles": [0.33, 0.67], "sensors": {}}
    for sensor, metrics in metrics_by_sensor.items():
        result["sensors"][sensor] = {}
        for key, values in metrics.items():
            result["sensors"][sensor][key] = [float(x) for x in np.quantile(values, [0.33, 0.67])]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def choose_enhancement(sensor: str, levels: dict[str, str]) -> str:
    if levels["noise_level"] == "high":
        return "speckle_denoise" if sensor == "sar" else "edge_preserving_denoise"
    if levels["contrast_level"] == "low" or levels["dynamic_range_level"] == "low":
        return "contrast_stretch"
    if levels["clarity_level"] == "low":
        return "mild_sharpen"
    return "none"


def infer(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    image_size = int(checkpoint.get("image_size", 224))
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(args.image) as im:
        tensor = tf(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().numpy()
    raw_scene = SCENES[int(probabilities.argmax())]
    confidence = float(probabilities.max())
    scene = raw_scene if confidence >= args.scene_threshold else "uncertain"

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    thresholds = calibration["sensors"][args.sensor]
    metrics = quality_metrics(args.image)
    levels = {
        "contrast_level": level(metrics["contrast_std"], thresholds["contrast_std"]),
        "dynamic_range_level": level(metrics["dynamic_range_p01_p99"], thresholds["dynamic_range_p01_p99"]),
        "clarity_level": level(metrics["sharpness_gradient"], thresholds["sharpness_gradient"]),
        "noise_level": level(metrics["high_frequency_noise"], thresholds["high_frequency_noise"]),
    }
    policy = dict(DEFAULT_POLICY[scene])
    policy["enhancement"] = choose_enhancement(args.sensor, levels)
    output = {
        "image": str(args.image.resolve()),
        "scene": {
            "label": scene,
            "raw_label": raw_scene,
            "confidence": round(confidence, 6),
            "probabilities": {name: round(float(prob), 6) for name, prob in zip(SCENES, probabilities)},
        },
        "environment": {
            "sensor_type": args.sensor,
            **levels,
            "raw_metrics": {k: round(v, 4) for k, v in metrics.items()},
        },
        "decision": policy,
        "description": (
            f"当前识别为{scene}场景（置信度{confidence:.3f}），"
            f"图像对比度{levels['contrast_level']}、清晰度{levels['clarity_level']}、"
            f"噪声{levels['noise_level']}，建议采用{policy['enhancement']}预处理和"
            f"{policy['detector_profile']}检测配置。"
        ),
        "warning": "决策阈值为首版实验默认值，必须通过目标检测消融实验后再定稿。",
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Scene cognition runtime")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate", help="calibrate quality thresholds from training images")
    c.add_argument("--train-csv", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    i = sub.add_parser("infer", help="infer scene, quality and decision")
    i.add_argument("--image", type=Path, required=True)
    i.add_argument("--sensor", choices=["ir", "sar"], required=True)
    i.add_argument("--checkpoint", type=Path, required=True)
    i.add_argument("--calibration", type=Path, required=True)
    i.add_argument("--scene-threshold", type=float, default=0.70)
    i.add_argument("--output", type=Path)
    i.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.command == "calibrate":
        calibrate(args.train_csv, args.output)
    else:
        infer(args)


if __name__ == "__main__":
    main()
