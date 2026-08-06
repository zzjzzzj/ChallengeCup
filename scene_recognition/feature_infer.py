from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from image_processing.feature_engineering import SCENES, extract_one


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "feature_baseline"


def selected_feature_values(
    extracted: dict[str, float], selected_features: list[str], precision: int = 9
) -> dict[str, float]:
    """Return the selected feature values in the same order as the metadata."""
    missing = [name for name in selected_features if name not in extracted]
    if missing:
        raise ValueError(f"提取结果缺少特征: {', '.join(missing)}")
    return {name: round(float(extracted[name]), precision) for name in selected_features}


def predict_scene_from_features(
    image: Path,
    model_path: Path,
    metadata_path: Path,
    *,
    features_only: bool = False,
    extracted: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Extract handcrafted features and optionally classify scene with the SVM model.

    Returns a dict compatible with the previous CLI JSON output. When
    ``features_only`` is True, classification fields are omitted.
    """

    if not image.is_file():
        raise FileNotFoundError(f"图片不存在或不是文件: {image}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"特征元数据不存在: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = extracted if extracted is not None else extract_one(image)
    selected_features = list(metadata["selected_features"])
    result: dict[str, Any] = {
        "image": str(image.resolve()),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_feature_values": selected_feature_values(features, selected_features),
        "extracted_features": {key: float(value) for key, value in features.items()},
        "metadata": metadata,
    }

    if features_only:
        return result

    if not model_path.is_file():
        raise FileNotFoundError(f"分类模型不存在: {model_path}")

    model = joblib.load(model_path)
    input_features = list(metadata["input_features"])
    missing = [name for name in input_features if name not in features]
    if missing:
        raise ValueError(f"提取结果缺少特征: {', '.join(missing)}")
    frame = pd.DataFrame([{name: features[name] for name in input_features}])
    probabilities = model.predict_proba(frame)[0]
    predicted_id = int(probabilities.argmax())
    scene_names = list(metadata.get("scene_names", SCENES))
    result.update(
        {
            "scene": scene_names[predicted_id],
            "confidence": round(float(probabilities[predicted_id]), 6),
            "probabilities": {
                name: round(float(value), 6) for name, value in zip(scene_names, probabilities)
            },
            "model": str(model_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="提取单张图片的入选特征值，并可同时进行场景分类")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_RUN_DIR / "scene_feature_svm.joblib",
        help="分类模型路径（默认使用项目自带模型）",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_RUN_DIR / "model_metadata.json",
        help="特征元数据路径（默认使用项目自带元数据）",
    )
    p.add_argument("--output", type=Path, help="可选的 JSON 结果保存路径")
    p.add_argument(
        "--features-only",
        action="store_true",
        help="只输出特征值，不加载分类模型",
    )
    args = p.parse_args()

    try:
        result = predict_scene_from_features(
            args.image,
            args.model,
            args.metadata,
            features_only=args.features_only,
        )
    except FileNotFoundError as exc:
        p.error(str(exc))

    # Keep CLI payload aligned with the previous public fields.
    cli_result = {
        "image": result["image"],
        "selected_feature_count": result["selected_feature_count"],
        "selected_features": result["selected_features"],
        "selected_feature_values": result["selected_feature_values"],
    }
    if not args.features_only:
        cli_result.update(
            {
                "scene": result["scene"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
        )

    text = json.dumps(cli_result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
