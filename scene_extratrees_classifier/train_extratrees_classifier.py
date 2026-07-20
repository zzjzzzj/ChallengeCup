from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from feature_engineering import FEATURE_NAMES, IMAGE_SIZE, SCENES, extract_one


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def path_tokens(path: Path) -> list[str]:
    return [token for token in TOKEN_SPLIT.split(path.as_posix().lower()) if token]


def infer_scene(path: Path) -> str | None:
    """从文件夹名或文件名中识别 air/sea/urban/forest。"""
    matches = [scene for scene in SCENES if scene in path_tokens(path)]
    return matches[0] if len(matches) == 1 else None


def infer_sensor(path: Path) -> str | None:
    """从路径中识别 ir/sar，仅用于分层划分，不作为模型输入。"""
    matches = [sensor for sensor in ("ir", "sar") if sensor in path_tokens(path)]
    return matches[0] if len(matches) == 1 else None


def discover_images(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_csv_image_path(value: str, csv_path: Path, data_root: Path | None) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    roots = [data_root, csv_path.parent] if data_root is not None else [csv_path.parent]
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (roots[0] / candidate).resolve()


def load_samples_from_csv(csv_path: Path, data_root: Path | None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    columns = {str(column).lower(): column for column in frame.columns}

    image_column = next(
        (columns[name] for name in ("image", "image_path", "path", "file") if name in columns),
        None,
    )
    scene_column = next(
        (columns[name] for name in ("scene", "label", "class") if name in columns),
        None,
    )
    sensor_column = next(
        (columns[name] for name in ("sensor", "modality") if name in columns),
        None,
    )

    if image_column is None or scene_column is None:
        raise ValueError(
            "CSV 必须包含图片列 image/image_path/path/file，以及场景列 scene/label/class"
        )

    samples = pd.DataFrame(
        {
            "image": [
                str(resolve_csv_image_path(str(value), csv_path, data_root))
                for value in frame[image_column]
            ],
            "scene": frame[scene_column].astype(str).str.lower().str.strip(),
        }
    )
    if sensor_column is not None:
        samples["sensor"] = frame[sensor_column].astype(str).str.lower().str.strip()
    else:
        samples["sensor"] = [infer_sensor(Path(value)) or "unknown" for value in samples["image"]]
    return samples


def load_samples_from_paths(data_root: Path) -> pd.DataFrame:
    images = discover_images(data_root)
    if not images:
        raise FileNotFoundError(f"在 {data_root} 下没有找到图片")

    rows: list[dict[str, str]] = []
    unresolved: list[Path] = []
    for image_path in images:
        relative_path = image_path.relative_to(data_root)
        scene = infer_scene(relative_path)
        if scene is None:
            unresolved.append(image_path)
            continue
        rows.append(
            {
                "image": str(image_path.resolve()),
                "scene": scene,
                "sensor": infer_sensor(relative_path) or "unknown",
            }
        )

    if unresolved:
        preview = "\n".join(f"  - {path}" for path in unresolved[:10])
        raise ValueError(
            "部分图片无法从路径中唯一识别 air/sea/urban/forest。\n"
            "请将场景名写入文件夹名/文件名，或使用 --labels-csv。\n"
            f"示例：\n{preview}"
        )
    return pd.DataFrame(rows)


def validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    samples["scene"] = samples["scene"].astype(str).str.lower().str.strip()
    samples["sensor"] = samples["sensor"].astype(str).str.lower().str.strip()

    invalid_scenes = sorted(set(samples["scene"]) - set(SCENES))
    if invalid_scenes:
        raise ValueError(f"发现不支持的场景标签: {invalid_scenes}；允许值为 {SCENES}")

    missing_files = [path for path in samples["image"] if not Path(path).is_file()]
    if missing_files:
        preview = "\n".join(f"  - {path}" for path in missing_files[:10])
        raise FileNotFoundError(f"存在找不到的图片：\n{preview}")

    scene_counts = samples["scene"].value_counts()
    missing_scenes = [scene for scene in SCENES if scene not in scene_counts]
    if missing_scenes:
        raise ValueError(f"训练数据缺少场景: {missing_scenes}")
    if int(scene_counts.min()) < 5:
        raise ValueError(f"每类至少需要 5 张图片，当前数量为: {scene_counts.to_dict()}")

    duplicates = samples["image"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError(f"发现重复图片路径，例如: {samples.loc[duplicates, 'image'].tolist()[:10]}")

    return samples.reset_index(drop=True)


def extract_feature_row(image_path: str) -> dict[str, float | str]:
    return {"image": image_path, **extract_one(Path(image_path))}


def build_feature_table(samples: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_feature_row)(path) for path in samples["image"]
    )
    features = pd.DataFrame(rows)
    table = samples.merge(features, on="image", how="left", validate="one_to_one")

    values = table[FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad_row, bad_col = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"出现无效特征值：第 {bad_row} 行，特征 {FEATURE_NAMES[int(bad_col)]}")
    return table


def choose_stratify_labels(samples: pd.DataFrame) -> pd.Series:
    """优先按场景+传感器分层，避免测试集中的 IR/SAR 比例偏移。"""
    combined = samples["scene"].astype(str) + "__" + samples["sensor"].astype(str)
    counts = combined.value_counts()
    if samples["sensor"].ne("unknown").all() and int(counts.min()) >= 2:
        return combined
    return samples["scene"]


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_base_model(seed: int, n_jobs: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 30 维灰度/纹理/频谱特征训练 ExtraTrees 四类场景分类器"
    )
    parser.add_argument("--data-root", type=Path, help="数据集根目录")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="逐图场景标签 CSV，包含 image 和 scene 两列，可选 sensor 列",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "extratrees",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=36,
        help="随机搜索参数组合数量，默认 36",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="不进行参数搜索，直接使用推荐的 ExtraTrees 参数",
    )
    args = parser.parse_args()

    if args.labels_csv is None and args.data_root is None:
        parser.error("必须提供 --data-root，或提供 --labels-csv")
    if args.labels_csv is not None and not args.labels_csv.is_file():
        parser.error(f"标签 CSV 不存在: {args.labels_csv}")
    if args.data_root is not None and not args.data_root.exists():
        parser.error(f"数据目录不存在: {args.data_root}")
    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size 应在 0.05 到 0.5 之间")
    if args.search_iterations < 1:
        parser.error("--search-iterations 必须大于等于 1")

    if args.labels_csv is not None:
        samples = load_samples_from_csv(args.labels_csv, args.data_root)
    else:
        samples = load_samples_from_paths(args.data_root)
    samples = validate_samples(samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("场景分布:", samples["scene"].value_counts().to_dict())
    print("传感器分布:", samples["sensor"].value_counts().to_dict())
    print(f"开始提取 {len(samples)} 张图片的 {len(FEATURE_NAMES)} 维特征……")

    table = build_feature_table(samples, args.n_jobs)
    table.to_csv(args.output_dir / "feature_table.csv", index=False, encoding="utf-8-sig")

    x = table[FEATURE_NAMES]
    y = table["scene"]
    stratify_labels = choose_stratify_labels(table)

    train_indices, test_indices = train_test_split(
        np.arange(len(table)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_labels,
    )
    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    base_model = make_base_model(args.seed, args.n_jobs)

    if args.no_search:
        model = base_model.fit(x_train, y_train)
        best_params = model.get_params()
        best_cv_f1 = None
    else:
        cv_folds = min(5, int(y_train.value_counts().min()))
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions={
                "n_estimators": [300, 500, 700, 900],
                "max_depth": [None, 10, 16, 24],
                "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 3, 4],
                "bootstrap": [False, True],
            },
            n_iter=args.search_iterations,
            scoring="f1_macro",
            cv=cv_folds,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            refit=True,
            verbose=2,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = float(search.best_score_)

    y_pred = model.predict(x_test)
    class_order = [str(value) for value in model.classes_]
    accuracy = float(np.mean(np.asarray(y_test) == np.asarray(y_pred)))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(
        y_test,
        y_pred,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=class_order)

    model_path = args.output_dir / "scene_feature_extratrees.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame(matrix, index=class_order, columns=class_order).to_csv(
        args.output_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(json_ready(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_table = table[["image", "scene", "sensor"]].copy()
    split_table["split"] = "train"
    split_table.loc[test_indices, "split"] = "test"
    split_table.to_csv(args.output_dir / "data_split.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "ExtraTreesClassifier",
        "scene_names": class_order,
        "input_features": FEATURE_NAMES,
        "selected_features": FEATURE_NAMES,
        "selected_feature_count": len(FEATURE_NAMES),
        "image_size": list(IMAGE_SIZE),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "scene_distribution": Counter(table["scene"]),
        "sensor_distribution": Counter(table["sensor"]),
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_f1,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n训练完成")
    print(f"模型: {model_path}")
    print(f"最佳参数: {best_params}")
    if best_cv_f1 is not None:
        print(f"交叉验证 Macro-F1: {best_cv_f1:.4f}")
    print(f"测试集 Accuracy: {accuracy:.4f}")
    print(f"测试集 Macro-F1: {macro_f1:.4f}")
    print("\n特征重要性前 10 名:")
    print(importance.head(10).to_string(index=False))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, labels=class_order, zero_division=0))


if __name__ == "__main__":
    main()
