from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from image_processing.feature_engineering import extract_image
from scene_recognition.feature_infer import selected_feature_values


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUN_DIR = PROJECT_ROOT / "scene_recognition" / "runs" / "feature_baseline"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SCENE_LABELS = {
    "air": "空中",
    "sea": "海面",
    "urban": "城市场景",
    "forest": "森林",
}

GROUP_LABELS = {
    "intensity": "灰度分布",
    "texture": "纹理结构",
    "frequency": "频域信息",
}

FEATURE_LABELS = {
    "int_std": "灰度标准差",
    "int_p50": "灰度中位数",
    "int_p75": "灰度 75% 分位数",
    "int_p90": "灰度 90% 分位数",
    "int_p95": "灰度 95% 分位数",
    "int_p99": "灰度 99% 分位数",
    "int_dynamic_range": "灰度动态范围",
    "int_entropy": "灰度熵",
    "tex_grad_p90": "梯度 90% 分位数",
    "tex_edge_density": "边缘密度",
    "tex_local_std_mean": "局部标准差均值",
    "tex_lbp_entropy": "LBP 纹理熵",
    "tex_glcm_d2_homogeneity": "GLCM 距离 2 同质性",
    "tex_glcm_d4_homogeneity": "GLCM 距离 4 同质性",
    "freq_spectral_entropy": "频谱熵",
}

FEATURE_DESCRIPTIONS = {
    "int_std": "衡量整幅图像灰度起伏程度，值越大通常意味着明暗变化越明显。",
    "int_dynamic_range": "灰度 99% 与 1% 分位数之差，可减少极端像素对范围判断的影响。",
    "int_entropy": "衡量灰度分布的不确定性与信息丰富程度。",
    "tex_grad_p90": "较强边缘位置的梯度水平，反映显著结构变化。",
    "tex_edge_density": "高于自适应阈值的梯度像素占比。",
    "tex_local_std_mean": "7×7 邻域内局部灰度波动的平均水平。",
    "tex_lbp_entropy": "局部二值模式分布的复杂程度。",
    "tex_glcm_d2_homogeneity": "距离 2 的灰度共生关系中，相似灰度对的集中程度。",
    "tex_glcm_d4_homogeneity": "距离 4 的灰度共生关系中，相似灰度对的集中程度。",
    "freq_spectral_entropy": "二维频谱能量分布的复杂程度。",
}


def feature_group(name: str) -> str:
    if name.startswith("int_"):
        return "intensity"
    if name.startswith("freq_"):
        return "frequency"
    return "texture"


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("tex_lbp_") and name[-2:].isdigit():
        return f"LBP 区间 {int(name[-2:]) + 1:02d} 占比"
    return name


def feature_description(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name.startswith("int_p"):
        return "图像灰度分布的稳健位置统计量。"
    if name.startswith("tex_lbp_"):
        return "该局部二值模式编码区间在全图中的出现比例。"
    return "特征工程筛选出的场景判别特征。"


@lru_cache(maxsize=1)
def load_resources() -> dict:
    metadata_path = RUN_DIR / "model_metadata.json"
    model_path = RUN_DIR / "scene_feature_svm.joblib"
    features_path = ARTIFACTS_DIR / "scene_features.csv"
    importance_path = RUN_DIR / "feature_importance.csv"

    required = [metadata_path, model_path, features_path, importance_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少运行文件：" + "；".join(missing))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = metadata["selected_features"]
    model = joblib.load(model_path)

    all_features = pd.read_csv(features_path, encoding="utf-8-sig")
    train_features = all_features[all_features["split"] == "train"]
    baseline = {
        name: {
            "mean": float(train_features[name].mean()),
            "std": float(train_features[name].std(ddof=0)),
        }
        for name in selected
    }
    importance_frame = pd.read_csv(importance_path, encoding="utf-8-sig")
    importance = {
        str(row.feature): float(row.importance)
        for row in importance_frame.itertuples(index=False)
    }
    return {
        "metadata": metadata,
        "model": model,
        "baseline": baseline,
        "importance": importance,
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.json.sort_keys = False

    @app.get("/")
    def index():
        return render_template("feature_dashboard.html")

    @app.get("/api/health")
    def health():
        try:
            resources = load_resources()
            return jsonify(
                {
                    "status": "ok",
                    "selected_feature_count": len(resources["metadata"]["selected_features"]),
                }
            )
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.post("/api/analyze")
    def analyze():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"message": "请选择一张图片后再开始分析。"}), 400

        started = time.perf_counter()
        try:
            with Image.open(upload.stream) as image:
                image.load()
                width, height = image.size
                image_format = image.format or "UNKNOWN"
                image_mode = image.mode
                extracted = extract_image(image)
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"message": "无法读取这张图片。请使用 PNG、JPG、BMP 或 TIFF 文件。"}), 400

        try:
            resources = load_resources()
            metadata = resources["metadata"]
            model = resources["model"]
            selected = metadata["selected_features"]
            values = selected_feature_values(extracted, selected)

            frame = pd.DataFrame(
                [{name: extracted[name] for name in metadata["input_features"]}]
            )
            probabilities = model.predict_proba(frame)[0]
            classes = [int(value) for value in model.classes_]
            scene_names = metadata["scene_names"]
            probability_map = {
                scene_names[class_id]: round(float(probability), 6)
                for class_id, probability in zip(classes, probabilities)
            }
            predicted_scene = max(probability_map, key=probability_map.get)

            feature_rows = []
            for index, name in enumerate(selected, start=1):
                value = values[name]
                base = resources["baseline"][name]
                z_score = 0.0
                if base["std"] > 1e-12:
                    z_score = (value - base["mean"]) / base["std"]
                group = feature_group(name)
                feature_rows.append(
                    {
                        "index": index,
                        "name": name,
                        "label": feature_label(name),
                        "description": feature_description(name),
                        "group": group,
                        "group_label": GROUP_LABELS[group],
                        "value": value,
                        "z_score": round(float(z_score), 4),
                        "importance": round(float(resources["importance"].get(name, 0.0)), 6),
                    }
                )
        except Exception as exc:
            app.logger.exception("Feature analysis failed")
            return jsonify({"message": f"分析未完成：{exc}"}), 500

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return jsonify(
            {
                "image": {
                    "name": upload.filename,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": image_mode,
                },
                "scene": predicted_scene,
                "scene_label": SCENE_LABELS.get(predicted_scene, predicted_scene),
                "confidence": probability_map[predicted_scene],
                "probabilities": [
                    {
                        "scene": name,
                        "label": SCENE_LABELS.get(name, name),
                        "value": probability_map.get(name, 0.0),
                    }
                    for name in scene_names
                ],
                "selected_feature_count": len(feature_rows),
                "features": feature_rows,
                "processing_ms": elapsed_ms,
                "baseline_note": "标准分数以训练集均值和标准差计算；0 表示接近训练集平均水平。",
            }
        )

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"message": "图片超过 16 MB，请压缩后再试。"}), 413

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动单图特征可视化分析台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    load_resources()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
