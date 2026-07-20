from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import FEATURE_NAMES, extract_one


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "extratrees"


def selected_feature_values(
    extracted: dict[str, float], selected_features: list[str], precision: int = 9
) -> dict[str, float]:
    missing = [name for name in selected_features if name not in extracted]
    if missing:
        raise ValueError(f"提取结果缺少特征: {', '.join(missing)}")
    return {name: round(float(extracted[name]), precision) for name in selected_features}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="提取单张图片特征，并使用 ExtraTrees 识别 air/sea/urban/forest"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_RUN_DIR / "scene_feature_extratrees.joblib",
        help="ExtraTrees 模型路径",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_RUN_DIR / "model_metadata.json",
        help="模型元数据路径",
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 保存路径")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="只输出特征，不加载分类模型",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"图片不存在或不是文件: {args.image}")

    metadata: dict = {}
    if args.metadata.is_file():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    elif not args.features_only:
        parser.error(f"模型元数据不存在: {args.metadata}")

    extracted = extract_one(args.image)
    selected_features = metadata.get("selected_features", FEATURE_NAMES)
    input_features = metadata.get("input_features", selected_features)

    result = {
        "image": str(args.image.resolve()),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_feature_values": selected_feature_values(extracted, selected_features),
    }

    if not args.features_only:
        if not args.model.is_file():
            parser.error(f"分类模型不存在: {args.model}")

        model = joblib.load(args.model)
        missing = [name for name in input_features if name not in extracted]
        if missing:
            parser.error(f"当前特征程序与模型不兼容，缺少: {missing}")

        frame = pd.DataFrame([{name: extracted[name] for name in input_features}])
        probabilities = model.predict_proba(frame)[0]
        class_names = [str(value) for value in model.classes_]
        best_index = int(probabilities.argmax())

        result.update(
            {
                "scene": class_names[best_index],
                "confidence": round(float(probabilities[best_index]), 6),
                "probabilities": {
                    name: round(float(value), 6)
                    for name, value in zip(class_names, probabilities)
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
