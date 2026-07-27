from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCENES = ["air", "sea", "urban", "forest"]
META_COLUMNS = ["image_path", "image_name", "sensor", "scene", "scene_id", "sequence_index", "split"]


def entropy_from_hist(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p /= max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def intensity_features(gray: np.ndarray) -> dict[str, float]:
    flat = gray.ravel()
    q = np.percentile(flat, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mean = float(flat.mean())
    std = float(flat.std())
    return {
        "int_mean": mean,
        "int_std": std,
        "int_min": float(flat.min()),
        "int_max": float(flat.max()),
        "int_p01": float(q[0]),
        "int_p05": float(q[1]),
        "int_p10": float(q[2]),
        "int_p25": float(q[3]),
        "int_p50": float(q[4]),
        "int_p75": float(q[5]),
        "int_p90": float(q[6]),
        "int_p95": float(q[7]),
        "int_p99": float(q[8]),
        "int_iqr": float(q[5] - q[3]),
        "int_dynamic_range": float(q[8] - q[0]),
        "int_cv": float(std / (mean + 1e-6)),
        "int_entropy": entropy_from_hist(flat),
        "int_skew": float(np.nan_to_num(stats.skew(flat), nan=0.0)),
        "int_kurtosis": float(np.nan_to_num(stats.kurtosis(flat), nan=0.0)),
        "int_dark_ratio": float((flat < 0.20).mean()),
        "int_bright_ratio": float((flat > 0.80).mean()),
    }


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    ]
    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    hist = np.bincount(code.ravel(), minlength=256).reshape(16, 16).sum(axis=1).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def glcm_for_offset(q: np.ndarray, dy: int, dx: int, levels: int = 16) -> np.ndarray:
    h, w = q.shape
    y1a, y1b = max(0, -dy), min(h, h - dy)
    x1a, x1b = max(0, -dx), min(w, w - dx)
    y2a, y2b = y1a + dy, y1b + dy
    x2a, x2b = x1a + dx, x1b + dx
    a = q[y1a:y1b, x1a:x1b].ravel()
    b = q[y2a:y2b, x2a:x2b].ravel()
    matrix = np.bincount(a * levels + b, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    matrix += matrix.T
    return matrix / max(matrix.sum(), 1.0)


def glcm_properties(p: np.ndarray) -> dict[str, float]:
    levels = p.shape[0]
    i, j = np.indices((levels, levels))
    diff = i - j
    mu_i = float((i * p).sum())
    mu_j = float((j * p).sum())
    sigma_i = math.sqrt(float((((i - mu_i) ** 2) * p).sum()))
    sigma_j = math.sqrt(float((((j - mu_j) ** 2) * p).sum()))
    return {
        "contrast": float((diff ** 2 * p).sum()),
        "dissimilarity": float((np.abs(diff) * p).sum()),
        "homogeneity": float((p / (1.0 + diff ** 2)).sum()),
        "energy": float((p ** 2).sum()),
        "correlation": float((((i - mu_i) * (j - mu_j) * p).sum()) / (sigma_i * sigma_j + 1e-9)),
    }


def texture_features(gray: np.ndarray) -> dict[str, float]:
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    grad = np.hypot(gx, gy)
    lap = ndimage.laplace(gray, mode="reflect")
    local_mean = ndimage.uniform_filter(gray, size=7, mode="reflect")
    local_sq = ndimage.uniform_filter(gray * gray, size=7, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    lbp = lbp_histogram(gray)
    result = {
        "tex_grad_mean": float(grad.mean()),
        "tex_grad_std": float(grad.std()),
        "tex_grad_p90": float(np.percentile(grad, 90)),
        "tex_edge_density": float((grad > (grad.mean() + grad.std())).mean()),
        "tex_horizontal_vertical_ratio": float((np.abs(gx).mean() + 1e-6) / (np.abs(gy).mean() + 1e-6)),
        "tex_laplacian_abs_mean": float(np.abs(lap).mean()),
        "tex_laplacian_var": float(lap.var()),
        "tex_local_std_mean": float(local_std.mean()),
        "tex_local_std_std": float(local_std.std()),
        "tex_lbp_entropy": float(-(lbp[lbp > 0] * np.log2(lbp[lbp > 0])).sum()),
    }
    result.update({f"tex_lbp_{i:02d}": float(v) for i, v in enumerate(lbp)})

    levels = 16
    q = np.minimum((gray * levels).astype(np.int32), levels - 1)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for distance in (1, 2, 4):
        props = []
        for dy, dx in directions:
            props.append(glcm_properties(glcm_for_offset(q, dy * distance, dx * distance, levels)))
        for key in props[0]:
            result[f"tex_glcm_d{distance}_{key}"] = float(np.mean([x[key] for x in props]))
    return result


def frequency_features(gray: np.ndarray) -> dict[str, float]:
    centered = gray - gray.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    radius = np.sqrt(((yy - h / 2) / h) ** 2 + ((xx - w / 2) / w) ** 2)
    power[h // 2, w // 2] = 0.0
    total = float(power.sum()) + 1e-12
    low = float(power[radius < 0.08].sum() / total)
    mid = float(power[(radius >= 0.08) & (radius < 0.20)].sum() / total)
    high = float(power[radius >= 0.20].sum() / total)
    p = power.ravel() / total
    p = p[p > 0]
    spectral_entropy = float(-(p * np.log2(p)).sum() / math.log2(power.size))
    weighted_radius = float((radius * power).sum() / total)
    return {
        "freq_low_energy": low,
        "freq_mid_energy": mid,
        "freq_high_energy": high,
        "freq_high_low_ratio": float(high / (low + 1e-9)),
        "freq_spectral_entropy": spectral_entropy,
        "freq_weighted_radius": weighted_radius,
    }


def extract_image(image: Image.Image) -> dict[str, float]:
    """Extract the complete handcrafted feature set from an opened image."""
    gray = np.asarray(
        image.convert("L").resize((128, 128), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return {**intensity_features(gray), **texture_features(gray), **frequency_features(gray)}


def extract_one(path: Path) -> dict[str, float]:
    with Image.open(path) as im:
        return extract_image(im)


def extract(index_csv: Path, output_csv: Path) -> None:
    rows = list(csv.DictReader(index_csv.open("r", encoding="utf-8-sig", newline="")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for n, row in enumerate(rows, start=1):
        features = extract_one(Path(row["image_path"]))
        output_rows.append(
            {
                **{key: row[key] for key in META_COLUMNS},
                "sensor_is_sar": int(row["sensor"] == "sar"),
                **{key: round(value, 9) for key, value in features.items()},
            }
        )
        if n % 100 == 0 or n == len(rows):
            print(f"features {n}/{len(rows)}")
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    catalog = {
        "intensity": [x for x in output_rows[0] if x.startswith("int_")],
        "texture": [x for x in output_rows[0] if x.startswith("tex_")],
        "frequency": [x for x in output_rows[0] if x.startswith("freq_")],
        "sensor_context": ["sensor_is_sar"],
    }
    output_csv.with_name("feature_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {
            name: float(value)
            for name, value in zip(SCENES, recall_score(y_true, y_pred, labels=range(4), average=None, zero_division=0))
        },
    }


def save_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *SCENES])
        for name, row in zip(SCENES, matrix.tolist()):
            writer.writerow([name, *row])


def evaluate(features_csv: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv, encoding="utf-8-sig")
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    visual_columns = [c for c in feature_columns if c != "sensor_is_sar"]
    groups = {
        "intensity": [c for c in visual_columns if c.startswith("int_")],
        "texture": [c for c in visual_columns if c.startswith("tex_")],
        "frequency": [c for c in visual_columns if c.startswith("freq_")],
    }
    groups["lbp"] = [c for c in visual_columns if c.startswith("tex_lbp_")]
    groups["glcm"] = [c for c in visual_columns if c.startswith("tex_glcm_")]
    groups["gradient_local"] = [
        c for c in visual_columns
        if c.startswith("tex_grad_") or c.startswith("tex_edge_") or c.startswith("tex_horizontal_")
        or c.startswith("tex_laplacian_") or c.startswith("tex_local_")
    ]
    groups["lbp_glcm"] = groups["lbp"] + groups["glcm"]
    groups["texture_frequency"] = groups["texture"] + groups["frequency"]
    groups["intensity_texture"] = groups["intensity"] + groups["texture"]
    groups["all_visual"] = visual_columns
    groups["all_visual_plus_sensor"] = visual_columns + ["sensor_is_sar"]

    train = df[df.split == "train"].copy()
    val = df[df.split == "val"].copy()
    test = df[df.split == "test"].copy()
    trainval = pd.concat([train, val], ignore_index=True)
    ablation_rows = []
    for name, columns in groups.items():
        model = make_pipeline(StandardScaler(), SVC(C=3.0, kernel="rbf", class_weight="balanced"))
        model.fit(train[columns], train.scene_id)
        val_pred = model.predict(val[columns])
        model.fit(trainval[columns], trainval.scene_id)
        test_pred = model.predict(test[columns])
        val_m = metric_dict(val.scene_id, val_pred)
        test_m = metric_dict(test.scene_id, test_pred)
        ablation_rows.append(
            {
                "feature_set": name,
                "feature_count": len(columns),
                "val_accuracy": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
                "test_accuracy": test_m["accuracy"],
                "test_macro_f1": test_m["macro_f1"],
            }
        )
    pd.DataFrame(ablation_rows).to_csv(output / "feature_ablation.csv", index=False, encoding="utf-8-sig")

    selection_rows = []
    best_k = None
    best_val_f1 = -1.0
    for k in (5, 10, 15, 20, 30, 40, 50, 60):
        model = make_pipeline(
            StandardScaler(), SelectKBest(score_func=f_classif, k=k), SVC(C=3.0, kernel="rbf", class_weight="balanced")
        )
        model.fit(train[visual_columns], train.scene_id)
        val_pred = model.predict(val[visual_columns])
        val_m = metric_dict(val.scene_id, val_pred)
        selection_rows.append({"k": k, "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"]})
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_k = k
    selected_model = make_pipeline(
        StandardScaler(), SelectKBest(score_func=f_classif, k=best_k),
        SVC(C=3.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    )
    selected_model.fit(trainval[visual_columns], trainval.scene_id)
    selected_test_pred = selected_model.predict(test[visual_columns])
    selected_test_m = metric_dict(test.scene_id, selected_test_pred)
    selection_rows.append(
        {
            "k": f"selected_{best_k}",
            "val_accuracy": None,
            "val_macro_f1": best_val_f1,
            "test_accuracy": selected_test_m["accuracy"],
            "test_macro_f1": selected_test_m["macro_f1"],
        }
    )
    pd.DataFrame(selection_rows).to_csv(output / "univariate_selection.csv", index=False, encoding="utf-8-sig")
    selector = selected_model.named_steps["selectkbest"]
    selected_features = [feature for feature, keep in zip(visual_columns, selector.get_support()) if keep]
    (output / "selected_features.json").write_text(
        json.dumps({"best_k": best_k, "features": selected_features, "test": selected_test_m}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(selected_model, output / "scene_feature_svm.joblib")
    save_confusion(output / "selected_confusion_matrix.csv", confusion_matrix(test.scene_id, selected_test_pred, labels=range(4)))
    selected_pred_df = test[META_COLUMNS].copy()
    selected_pred_df["predicted_scene_id"] = selected_test_pred
    selected_pred_df["predicted_scene"] = [SCENES[int(x)] for x in selected_test_pred]
    selected_pred_df["correct"] = selected_pred_df.scene_id == selected_pred_df.predicted_scene_id
    selected_pred_df.to_csv(output / "selected_test_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "model_metadata.json").write_text(
        json.dumps(
            {
                "model": "StandardScaler + ANOVA SelectKBest + RBF-SVM",
                "input_features": visual_columns,
                "selected_features": selected_features,
                "scene_names": SCENES,
                "test_metrics": selected_test_m,
                "warning": "joblib models must only be loaded from trusted local sources",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    classifiers = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=3.0, kernel="rbf", class_weight="balanced", probability=True)),
        "random_forest": RandomForestClassifier(
            n_estimators=600, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    results = {}
    predictions = {}
    for name, model in classifiers.items():
        model.fit(trainval[visual_columns], trainval.scene_id)
        pred = model.predict(test[visual_columns])
        results[name] = metric_dict(test.scene_id, pred)
        predictions[name] = pred
        results[name]["classification_report"] = classification_report(
            test.scene_id, pred, labels=range(4), target_names=SCENES, zero_division=0, output_dict=True
        )
        for sensor in ("ir", "sar"):
            mask = test.sensor == sensor
            results[name][f"{sensor}_accuracy"] = float(accuracy_score(test.loc[mask, "scene_id"], pred[mask.to_numpy()]))
            results[name][f"{sensor}_macro_f1"] = float(
                f1_score(test.loc[mask, "scene_id"], pred[mask.to_numpy()], average="macro", zero_division=0)
            )

    best_name = max(results, key=lambda x: results[x]["macro_f1"])
    best_pred = predictions[best_name]
    results["best_model"] = best_name
    results["split_note"] = "train+val fitting, sequentially isolated test set"
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion(output / "confusion_matrix.csv", confusion_matrix(test.scene_id, best_pred, labels=range(4)))
    pred_df = test[META_COLUMNS].copy()
    pred_df["predicted_scene_id"] = best_pred
    pred_df["predicted_scene"] = [SCENES[int(x)] for x in best_pred]
    pred_df["correct"] = pred_df.scene_id == pred_df.predicted_scene_id
    pred_df.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")

    forest = classifiers["random_forest"]
    importances = pd.DataFrame({"feature": visual_columns, "importance": forest.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importances.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")

    stats_rows = []
    for (sensor, scene), group in df.groupby(["sensor", "scene"]):
        for feature in visual_columns:
            stats_rows.append(
                {
                    "sensor": sensor,
                    "scene": scene,
                    "feature": feature,
                    "mean": float(group[feature].mean()),
                    "std": float(group[feature].std(ddof=0)),
                }
            )
    pd.DataFrame(stats_rows).to_csv(output / "feature_stats_by_sensor_scene.csv", index=False, encoding="utf-8-sig")

    signatures = {}
    for sensor in ("ir", "sar"):
        sensor_df = df[df.sensor == sensor]
        signatures[sensor] = {}
        for scene in sorted(sensor_df.scene.unique()):
            current = sensor_df[sensor_df.scene == scene]
            other = sensor_df[sensor_df.scene != scene]
            scored = []
            for feature in visual_columns:
                delta = float(current[feature].mean() - other[feature].mean())
                pooled = float(np.sqrt((current[feature].var(ddof=0) + other[feature].var(ddof=0)) / 2.0))
                effect = delta / (pooled + 1e-9)
                scored.append({"feature": feature, "effect_size": effect, "direction": "higher" if effect > 0 else "lower"})
            signatures[sensor][scene] = sorted(scored, key=lambda x: abs(x["effect_size"]), reverse=True)[:10]
    (output / "scene_feature_signatures.json").write_text(
        json.dumps(signatures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = importances.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top.feature, top.importance, color="#2E74B5")
    plt.xlabel("Random forest importance")
    plt.tight_layout()
    plt.savefig(output / "top_feature_importance.png", dpi=180)
    plt.close()

    cm = confusion_matrix(test.scene_id, selected_test_pred, labels=range(4))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(4), SCENES)
    plt.yticks(range(4), SCENES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()

    pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=42))
    coords = pca_pipe.fit_transform(df[visual_columns])
    plt.figure(figsize=(8, 6))
    colors = {"air": "#2E74B5", "sea": "#17A2B8", "urban": "#B07D16", "forest": "#2E8B57"}
    markers = {"ir": "o", "sar": "^"}
    for scene in SCENES:
        for sensor in ("ir", "sar"):
            mask = (df.scene == scene) & (df.sensor == sensor)
            if mask.any():
                plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, c=colors[scene], marker=markers[sensor], label=f"{scene}-{sensor}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output / "pca_scene_sensor.png", dpi=180)
    plt.close()
    print(json.dumps({"ablation": ablation_rows, "models": results}, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Scene feature engineering")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--index", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("evaluate")
    v.add_argument("--features", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "extract":
        extract(args.index, args.output)
    else:
        evaluate(args.features, args.output)


if __name__ == "__main__":
    main()
