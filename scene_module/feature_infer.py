from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import SCENES, extract_one


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "feature_baseline"


def selected_feature_values(
    extracted: dict[str, float], selected_features: list[str], precision: int = 9
) -> dict[str, float]:
    """Return the selected feature values in the same order as the metadata."""
    missing = [name for name in selected_features if name not in extracted]
    if missing:
        raise ValueError(f"提取结果缺少特征: {', '.join(missing)}")
    return {name: round(float(extracted[name]), precision) for name in selected_features}


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

    if not args.image.is_file():
        p.error(f"图片不存在或不是文件: {args.image}")
    if not args.metadata.is_file():
        p.error(f"特征元数据不存在: {args.metadata}")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    extracted = extract_one(args.image)
    selected_features = metadata["selected_features"]
    result = {
        "image": str(args.image.resolve()),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_feature_values": selected_feature_values(extracted, selected_features),
    }

    if not args.features_only:
        if not args.model.is_file():
            p.error(f"分类模型不存在: {args.model}")
        model = joblib.load(args.model)
        frame = pd.DataFrame([{name: extracted[name] for name in metadata["input_features"]}])
        probabilities = model.predict_proba(frame)[0]
        predicted_id = int(probabilities.argmax())
        scene_names = metadata.get("scene_names", SCENES)
        result.update(
            {
                "scene": scene_names[predicted_id],
                "confidence": round(float(probabilities[predicted_id]), 6),
                "probabilities": {
                    name: round(float(value), 6)
                    for name, value in zip(scene_names, probabilities)
                },
            }
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
