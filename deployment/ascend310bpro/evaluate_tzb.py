"""Evaluate the four indicators required by the ChallengeCup test protocol.

The evaluator is deliberately board-side and dependency-light.  It consumes
the JSONL files written by ``infer_best_6class_npu.py`` or
``routed_infer_npu.py`` and YOLO-format test labels, so the same predictions
used for Ascend 310B evidence are used for mAP, KRR, and New-mAP.

Protocol:
* the base test set is evaluated by both the pre-increment and post-increment
  models;
* KRR is post-increment old-class mAP divided by pre-increment old-class mAP;
* the incremental test set is evaluated by the post-increment model and its
  new-class mAP is New-mAP;
* FPS is read from the inference summary, or may be supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - exercised on a missing board dependency
    yaml = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised on a missing board dependency
    Image = None  # type: ignore[assignment,misc]


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_CLASSES = Path(__file__).resolve().parent / "classes_6.txt"
DEFAULT_OUTPUT = Path("outputs/ascend310bpro_tzb_metrics/metrics.json")

Box = Tuple[float, float, float, float]
GroundTruth = Dict[str, List[Tuple[Box, int]]]


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def read_classes(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names:
        raise ValueError("classes file is empty: %s" % path)
    return names


def _resolve_relative(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _split_entries(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def read_dataset_images(value: Path) -> List[Path]:
    """Read a directory, image list, or YOLO data YAML into an image list."""

    value = value.expanduser().resolve()
    if value.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("evaluate_tzb.py requires PyYAML; install requirements-runtime.txt")
        config = yaml.safe_load(value.read_text(encoding="utf-8")) or {}
        root = _resolve_relative(str(config.get("path", ".")), value.parent)
        split_value = config.get("test") or config.get("val")
        if not split_value:
            raise ValueError("dataset YAML must define test or val: %s" % value)
        entries = _split_entries(split_value)
        paths: List[Path] = []
        for entry in entries:
            candidate = _resolve_relative(entry, root)
            if candidate.suffix.lower() in {".txt", ".lst", ".list"}:
                paths.extend(_read_image_list(candidate, root))
            elif candidate.is_dir():
                paths.extend(_iter_images(candidate))
            elif candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(candidate)
            else:
                raise FileNotFoundError("dataset split not found: %s" % candidate)
        return _unique_paths(paths)
    if value.suffix.lower() in {".txt", ".lst", ".list"}:
        return _unique_paths(_read_image_list(value, value.parent))
    if value.is_dir():
        return _iter_images(value)
    if value.is_file() and value.suffix.lower() in IMAGE_EXTENSIONS:
        return [value]
    raise FileNotFoundError("dataset input not found: %s" % value)


def _iter_images(directory: Path) -> List[Path]:
    return sorted(
        (path.resolve() for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )


def _read_image_list(path: Path, base: Path) -> List[Path]:
    rows = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = _resolve_relative(line, base)
        if not candidate.is_file():
            raise FileNotFoundError("image in list not found: %s" % candidate)
        rows.append(candidate)
    return rows


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique = {path.resolve() for path in paths}
    return sorted(unique, key=lambda path: path.as_posix().lower())


def _prediction_keys(path: str) -> List[str]:
    raw = Path(path).expanduser()
    keys = [str(raw.resolve()) if raw.is_absolute() else str(raw.resolve())]
    keys.extend([raw.name.lower(), raw.stem.lower()])
    return keys


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at %s:%d" % (path, line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError("prediction row must be an object at %s:%d" % (path, line_number))
            rows.append(row)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else payload.get("predictions", [])
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image = str(row.get("image", "")).strip()
        if not image:
            raise ValueError("prediction row has no image: %s" % path)
        for key in _prediction_keys(image):
            result[key] = row
    return result


def _find_prediction(predictions: Dict[str, Dict[str, Any]], image: Path) -> Optional[Dict[str, Any]]:
    for key in _prediction_keys(str(image)):
        if key in predictions:
            return predictions[key]
    return None


def _label_candidates(image: Path, labels_root: Optional[Path]) -> List[Path]:
    candidates = [image.with_suffix(".txt")]
    parts = list(image.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        candidates.append(Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt"))
    if labels_root is not None:
        if "images" in parts:
            index = len(parts) - 1 - parts[::-1].index("images")
            candidates.insert(0, labels_root / Path(*parts[index + 1 :]).with_suffix(".txt"))
        candidates.insert(0, labels_root / image.name.replace(image.suffix, ".txt"))
        candidates.insert(0, labels_root / (image.stem + ".txt"))
    return candidates


def find_label(image: Path, labels_root: Optional[Path]) -> Optional[Path]:
    for candidate in _label_candidates(image, labels_root):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _image_size(image: Path, prediction: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    size = prediction.get("image_size") if prediction else None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        width, height = int(size[0]), int(size[1])
        if width > 0 and height > 0:
            return width, height
    if Image is None:
        raise RuntimeError("evaluate_tzb.py requires Pillow; install requirements-runtime.txt")
    with Image.open(image) as handle:
        return int(handle.size[0]), int(handle.size[1])


def load_ground_truth(images: Sequence[Path], predictions: Dict[str, Dict[str, Any]], labels_root: Optional[Path]) -> GroundTruth:
    ground_truth: GroundTruth = {}
    missing: List[str] = []
    for image in images:
        label = find_label(image, labels_root)
        if label is None:
            missing.append(str(image))
            continue
        width, height = _image_size(image, _find_prediction(predictions, image))
        boxes: List[Tuple[Box, int]] = []
        for line_number, raw in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
            values = raw.strip().split()
            if not values:
                continue
            if len(values) < 5:
                raise ValueError("invalid YOLO label at %s:%d" % (label, line_number))
            class_id = int(float(values[0]))
            x_center, y_center, box_width, box_height = (float(item) for item in values[1:5])
            x1 = (x_center - box_width / 2.0) * width
            y1 = (y_center - box_height / 2.0) * height
            x2 = (x_center + box_width / 2.0) * width
            y2 = (y_center + box_height / 2.0) * height
            boxes.append((_clip_box((x1, y1, x2, y2), width, height), class_id))
        ground_truth[str(image.resolve())] = boxes
    if missing:
        preview = "\n".join("  " + item for item in missing[:10])
        raise FileNotFoundError("missing YOLO labels for %d images:\n%s" % (len(missing), preview))
    return ground_truth


def _clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)), max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))


def _prediction_boxes(row: Optional[Dict[str, Any]]) -> List[Tuple[Box, float, int]]:
    if not row:
        return []
    result = []
    for item in row.get("detections", []) or []:
        box = item.get("box") or item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        score = _float(item.get("score"), 0.0) or 0.0
        class_id = item.get("class_id")
        if class_id is None:
            continue
        result.append(((float(box[0]), float(box[1]), float(box[2]), float(box[3])), score, int(class_id)))
    return result


def iou(left: Box, right: Box) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(predictions: Sequence[Tuple[str, Box, float]], ground_truth: GroundTruth, class_id: int, iou_threshold: float) -> Optional[float]:
    gt_by_image: Dict[str, List[Box]] = {}
    for image_key, boxes in ground_truth.items():
        gt_by_image[image_key] = [box for box, label in boxes if label == class_id]
    support = sum(len(boxes) for boxes in gt_by_image.values())
    if support == 0:
        return None
    ordered = sorted(predictions, key=lambda item: item[2], reverse=True)
    matched: Dict[str, set[int]] = defaultdict(set)
    true_positive: List[int] = []
    false_positive: List[int] = []
    for image_key, predicted_box, _score in ordered:
        candidates = gt_by_image.get(image_key, [])
        best_index = -1
        best_iou = 0.0
        for index, target_box in enumerate(candidates):
            if index in matched[image_key]:
                continue
            overlap = iou(predicted_box, target_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[image_key].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    if not true_positive:
        return 0.0
    cumulative_tp = 0
    cumulative_fp = 0
    precisions: List[float] = []
    recalls: List[float] = []
    for tp, fp in zip(true_positive, false_positive):
        cumulative_tp += tp
        cumulative_fp += fp
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
        recalls.append(cumulative_tp / support)
    sampled = []
    for threshold in (index / 100.0 for index in range(101)):
        eligible = [precision for precision, recall in zip(precisions, recalls) if recall >= threshold]
        sampled.append(max(eligible) if eligible else 0.0)
    return sum(sampled) / len(sampled)


def evaluate_predictions(images: Sequence[Path], predictions: Dict[str, Dict[str, Any]], ground_truth: GroundTruth, class_names: Sequence[str], selected_class_ids: Sequence[int]) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = {}
    for class_id in selected_class_ids:
        class_name = class_names[class_id]
        rows: List[Tuple[str, Box, float]] = []
        for image in images:
            key = str(image.resolve())
            for box, score, predicted_class_id in _prediction_boxes(_find_prediction(predictions, image)):
                if predicted_class_id == class_id:
                    rows.append((key, box, score))
        support = sum(1 for boxes in ground_truth.values() for _box, label in boxes if label == class_id)
        ap50 = average_precision(rows, ground_truth, class_id, 0.50)
        ap5095_values = [average_precision(rows, ground_truth, class_id, threshold) for threshold in (0.50 + 0.05 * index for index in range(10))]
        valid = [value for value in ap5095_values if value is not None]
        per_class[class_name] = {
            "class_id": class_id,
            "support": support,
            "map50": ap50,
            "map50_95": sum(valid) / len(valid) if valid else None,
        }
    supported = [row for row in per_class.values() if row["support"] > 0]
    return {
        "images": len(images),
        "classes": [class_names[class_id] for class_id in selected_class_ids],
        "per_class": per_class,
        "map50": sum(row["map50"] for row in supported if row["map50"] is not None) / len(supported) if supported else None,
        "map50_95": sum(row["map50_95"] for row in supported if row["map50_95"] is not None) / len(supported) if supported else None,
        "missing_support": [class_names[class_id] for class_id in selected_class_ids if per_class[class_names[class_id]]["support"] == 0],
    }


def _class_ids(class_names: Sequence[str], value: str) -> List[int]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in class_names]
    if unknown:
        raise ValueError("unknown classes: %s" % ", ".join(unknown))
    return [class_names.index(name) for name in names]


def _fps_from_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"value": None, "unit": "images/s", "source": None, "definition": "not supplied"}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    direct = _float(payload.get("fps"))
    if direct is not None:
        return {"value": direct, "unit": "images/s", "source": str(path.resolve()), "definition": payload.get("fps_definition", "inference summary")}
    end_to_end = _float(payload.get("fps", {}).get("end_to_end") if isinstance(payload.get("fps"), dict) else None)
    if end_to_end is not None:
        return {"value": end_to_end, "unit": "images/s", "source": str(path.resolve()), "definition": payload.get("fps_definition", "end-to-end inference")}
    avg = _float(payload.get("avg_end_to_end_ms"))
    if avg and avg > 0:
        return {"value": 1000.0 / avg, "unit": "images/s", "source": str(path.resolve()), "definition": "1000 / avg_end_to_end_ms"}
    raise ValueError("FPS not found in inference summary: %s" % path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mAP, KRR, New-mAP and FPS for the ChallengeCup test protocol.")
    parser.add_argument("--base-data", type=Path, required=True, help="base test directory/list/YAML")
    parser.add_argument("--new-data", type=Path, required=True, help="incremental/new-class test directory/list/YAML")
    parser.add_argument("--before-predictions", type=Path, required=True, help="pre-increment model predictions on base test")
    parser.add_argument("--after-base-predictions", type=Path, required=True, help="post-increment model predictions on base test")
    parser.add_argument("--after-new-predictions", type=Path, required=True, help="post-increment model predictions on incremental test")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--base-labels", type=Path, help="optional labels directory for base data")
    parser.add_argument("--new-labels", type=Path, help="optional labels directory for new data")
    parser.add_argument("--old-classes", default="soldier,small_aircraft,warship,tank")
    parser.add_argument("--new-classes", default="patrol_boat,armored_vehicle")
    parser.add_argument("--fps-summary", type=Path, help="summary.json produced by the board inference")
    parser.add_argument("--fps", type=float, help="explicit FPS when a summary is unavailable")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    class_names = read_classes(args.classes.resolve())
    old_ids = _class_ids(class_names, args.old_classes)
    new_ids = _class_ids(class_names, args.new_classes)
    base_images = read_dataset_images(args.base_data)
    new_images = read_dataset_images(args.new_data)
    before_predictions = load_predictions(args.before_predictions.resolve())
    after_base_predictions = load_predictions(args.after_base_predictions.resolve())
    after_new_predictions = load_predictions(args.after_new_predictions.resolve())
    base_gt = load_ground_truth(base_images, before_predictions, args.base_labels.resolve() if args.base_labels else None)
    new_gt = load_ground_truth(new_images, after_new_predictions, args.new_labels.resolve() if args.new_labels else None)

    before_base = evaluate_predictions(base_images, before_predictions, base_gt, class_names, old_ids)
    after_base = evaluate_predictions(base_images, after_base_predictions, base_gt, class_names, old_ids)
    after_new = evaluate_predictions(new_images, after_new_predictions, new_gt, class_names, new_ids)
    old_before = before_base["map50"]
    old_after = after_base["map50"]
    krr50 = old_after / old_before if old_before not in (None, 0) and old_after is not None else None
    old_before_95 = before_base["map50_95"]
    old_after_95 = after_base["map50_95"]
    krr95 = old_after_95 / old_before_95 if old_before_95 not in (None, 0) and old_after_95 is not None else None
    fps = {"value": args.fps, "unit": "images/s", "source": "--fps", "definition": "user supplied"} if args.fps is not None else _fps_from_summary(args.fps_summary)

    report = {
        "protocol": "challengecup-tzb-v1",
        "protocol_notes": {
            "base_model_pair_same_test_set": True,
            "new_model_new_class_test_set": True,
            "krr_definition": "old mAP after / old mAP before",
            "new_map_definition": "mAP of new classes on the incremental test set",
            "fps_definition": fps["definition"],
        },
        "class_order": class_names,
        "old_classes": [class_names[index] for index in old_ids],
        "new_classes": [class_names[index] for index in new_ids],
        "datasets": {"base": [str(path) for path in base_images], "new": [str(path) for path in new_images]},
        "metrics": {
            "base_detection": {"before": before_base, "after": after_base},
            "new_detection": {"after": after_new},
            "krr": {"map50": krr50, "map50_95": krr95, "old_map_before": old_before, "old_map_after": old_after},
            "new_map": {"map50": after_new["map50"], "map50_95": after_new["map50_95"]},
            "fps": fps,
        },
        "scorecard": {
            "mAP": after_base["map50"],
            "mAP@0.5": after_base["map50"],
            "mAP-before": before_base["map50"],
            "mAP-after": after_base["map50"],
            "KRR": krr50,
            "New-mAP": after_new["map50"],
            "FPS": fps["value"],
        },
        "evaluation_ready": not (before_base["missing_support"] or after_base["missing_support"] or after_new["missing_support"] or fps["value"] is None),
        "score_targets": {"mAP@0.5": 0.80, "New-mAP": 0.60, "KRR": 0.95, "FPS": 30.0},
        "privacy": {"network_used": False, "local_predictions_and_labels_only": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["mAP@0.5", report["scorecard"]["mAP@0.5"], "ratio"])
        writer.writerow(["KRR", report["scorecard"]["KRR"], "ratio"])
        writer.writerow(["New-mAP", report["scorecard"]["New-mAP"], "ratio"])
        writer.writerow(["FPS", report["scorecard"]["FPS"], "images/s"])
    print(json.dumps({"scorecard": report["scorecard"], "evaluation_ready": report["evaluation_ready"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print("evaluate_tzb.py: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
