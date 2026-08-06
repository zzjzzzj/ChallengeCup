from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

from scene_recognition.train_scene_classifier import SCENES, build_model


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

# Fixed thresholds used when no per-sensor calibration file is available.
DEFAULT_QUALITY_THRESHOLDS = {
    "contrast_std": [30.0, 65.0],
    "dynamic_range_p01_p99": [80.0, 160.0],
    "sharpness_gradient": [4.5, 12.0],
    "high_frequency_noise": [2.5, 8.0],
    "colorfulness": [8.0, 24.0],
}

NUMERIC_METRIC_KEYS = (
    "mean_gray",
    "contrast_std",
    "dynamic_range_p01_p99",
    "p05_gray",
    "p50_gray",
    "p95_gray",
    "sharpness_gradient",
    "high_frequency_noise",
    "edge_density",
    "colorfulness",
    "aspect_ratio",
    "width",
    "height",
)


def quality_metrics(path: Path) -> dict[str, float | int | str]:
    """Compute image quality metrics used by Agent and scene runtime."""

    with Image.open(path) as opened:
        opened.load()
        rgb = opened.convert("RGB")
        gray_image = opened.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32)
        rgb_values = np.asarray(rgb, dtype=np.float32)
        blurred = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
        width, height = opened.size
        mode = opened.mode

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    residual = gray - blurred
    q01, q05, q50, q95, q99 = np.percentile(gray, [1, 5, 50, 95, 99])
    rg = rgb_values[..., 0] - rgb_values[..., 1]
    yb = 0.5 * (rgb_values[..., 0] + rgb_values[..., 1]) - rgb_values[..., 2]
    colorfulness = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2) + 0.3 * math.sqrt(
        float(rg.mean()) ** 2 + float(yb.mean()) ** 2
    )
    edge_strength = (float(np.abs(gx).mean()) + float(np.abs(gy).mean())) / 2.0
    return {
        "width": int(width),
        "height": int(height),
        "mode": str(mode),
        "mean_gray": round(float(gray.mean()), 6),
        "contrast_std": round(float(gray.std()), 6),
        "dynamic_range_p01_p99": round(float(q99 - q01), 6),
        "p05_gray": round(float(q05), 6),
        "p50_gray": round(float(q50), 6),
        "p95_gray": round(float(q95), 6),
        "sharpness_gradient": round(edge_strength, 6),
        "high_frequency_noise": round(float(np.mean(np.abs(residual))), 6),
        "edge_density": round(
            float((np.abs(gx).mean(axis=0).mean() + np.abs(gy).mean(axis=1).mean()) / 255.0), 6
        ),
        "colorfulness": round(float(colorfulness), 6),
        "aspect_ratio": round(float(width / max(1, height)), 6),
    }


def level(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    return "high"


def quality_levels(
    metrics: dict[str, Any],
    thresholds: dict[str, list[float]] | None = None,
) -> dict[str, str]:
    """Map raw metrics to low/medium/high levels."""

    active = thresholds or DEFAULT_QUALITY_THRESHOLDS
    return {
        "contrast_level": level(float(metrics["contrast_std"]), active["contrast_std"]),
        "dynamic_range_level": level(
            float(metrics["dynamic_range_p01_p99"]), active["dynamic_range_p01_p99"]
        ),
        "clarity_level": level(float(metrics["sharpness_gradient"]), active["sharpness_gradient"]),
        "noise_level": level(float(metrics["high_frequency_noise"]), active["high_frequency_noise"]),
        "color_level": level(float(metrics.get("colorfulness", 0.0)), active.get("colorfulness", [8.0, 24.0])),
    }


def calibrate(train_csv: Path, output: Path) -> None:
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8-sig", newline="")))
    metrics_by_sensor: dict[str, dict[str, list[float]]] = {}
    for idx, row in enumerate(rows, start=1):
        metrics = quality_metrics(Path(row["image_path"]))
        sensor = row["sensor"]
        metrics_by_sensor.setdefault(sensor, {key: [] for key in NUMERIC_METRIC_KEYS})
        for key in NUMERIC_METRIC_KEYS:
            metrics_by_sensor[sensor][key].append(float(metrics[key]))
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


def build_environment_and_policy(
    image: Path,
    sensor: str,
    scene_label: str,
    scene_confidence: float,
    *,
    calibration: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build environment state and detector policy for a predicted scene."""

    metrics = metrics or quality_metrics(image)
    if calibration and sensor in calibration.get("sensors", {}):
        sensor_thresholds = calibration["sensors"][sensor]
        thresholds = {
            key: list(sensor_thresholds[key])
            for key in (
                "contrast_std",
                "dynamic_range_p01_p99",
                "sharpness_gradient",
                "high_frequency_noise",
            )
            if key in sensor_thresholds
        }
        if "colorfulness" in sensor_thresholds:
            thresholds["colorfulness"] = list(sensor_thresholds["colorfulness"])
        else:
            thresholds["colorfulness"] = DEFAULT_QUALITY_THRESHOLDS["colorfulness"]
    else:
        thresholds = DEFAULT_QUALITY_THRESHOLDS

    levels = quality_levels(metrics, thresholds)
    policy_key = scene_label if scene_label in DEFAULT_POLICY else "uncertain"
    policy = dict(DEFAULT_POLICY[policy_key])
    policy["enhancement"] = choose_enhancement(sensor, levels)
    return {
        "environment": {
            "sensor_type": sensor,
            "scene_label": scene_label,
            "scene_confidence": scene_confidence,
            **levels,
            "raw_metrics": metrics,
            "state_vector": {
                "scene_confidence": round(float(scene_confidence), 6),
                "contrast_norm": round(min(float(metrics["contrast_std"]) / 100.0, 1.0), 6),
                "clarity_norm": round(min(float(metrics["sharpness_gradient"]) / 18.0, 1.0), 6),
                "noise_norm": round(min(float(metrics["high_frequency_noise"]) / 12.0, 1.0), 6),
                "color_norm": round(min(float(metrics.get("colorfulness", 0.0)) / 35.0, 1.0), 6),
            },
        },
        "decision": policy,
        "levels": levels,
        "enhancement": policy["enhancement"],
    }


def predict_scene_cnn(
    image: Path,
    checkpoint_path: Path,
    *,
    scene_threshold: float = 0.70,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run the ResNet/CNN scene classifier trained by train_scene_classifier."""

    if not image.is_file():
        raise FileNotFoundError(f"图片不存在: {image}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"场景CNN检查点不存在: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image) as opened:
        tensor = transform(opened.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].detach().cpu().numpy()
    raw_scene = SCENES[int(probabilities.argmax())]
    confidence = float(probabilities.max())
    scene = raw_scene if confidence >= scene_threshold else "uncertain"
    return {
        "label": scene,
        "raw_label": raw_scene,
        "confidence": round(confidence, 6),
        "probabilities": {
            name: round(float(prob), 6) for name, prob in zip(SCENES, probabilities)
        },
        "source": "scene_cnn_model",
        "checkpoint": str(checkpoint_path.resolve()),
    }


def infer(args: argparse.Namespace) -> None:
    scene_result = predict_scene_cnn(
        args.image,
        args.checkpoint,
        scene_threshold=args.scene_threshold,
        device_name="cpu" if args.cpu else "auto",
    )
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    bundled = build_environment_and_policy(
        args.image,
        args.sensor,
        scene_result["label"],
        scene_result["confidence"],
        calibration=calibration,
    )
    levels = bundled["levels"]
    policy = bundled["decision"]
    output = {
        "image": str(args.image.resolve()),
        "scene": {
            "label": scene_result["label"],
            "raw_label": scene_result["raw_label"],
            "confidence": scene_result["confidence"],
            "probabilities": scene_result["probabilities"],
        },
        "environment": {
            "sensor_type": args.sensor,
            **{key: levels[key] for key in levels if key.endswith("_level")},
            "raw_metrics": {
                key: round(float(value), 4)
                for key, value in bundled["environment"]["raw_metrics"].items()
                if isinstance(value, (int, float))
            },
        },
        "decision": policy,
        "description": (
            f"当前识别为{scene_result['label']}场景（置信度{scene_result['confidence']:.3f}），"
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
