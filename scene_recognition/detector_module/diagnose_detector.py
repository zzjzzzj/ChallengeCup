from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

from scene_recognition.detector_module import CLASS_NAMES
from scene_recognition.detector_module.boxes import YoloBox, box_iou, parse_yolo_boxes, size_bucket
from scene_recognition.detector_module.dataset import DetectionSample, read_scene_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = PROJECT_ROOT / "image_processing" / "artifacts" / "scene_index.csv"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "detector_module"
    / "runs"
    / "yolov8n_baseline_v1"
    / "weights"
    / "submission_map50.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose detector weak spots by class, sensor, scene, and object size."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def read_ground_truth(
    samples: list[DetectionSample], class_names: list[str]
) -> dict[str, list[YoloBox]]:
    return {
        str(sample.image_path.resolve()): parse_yolo_boxes(sample.label_path, len(class_names))
        for sample in samples
    }


def collect_predictions(
    model_path: Path,
    samples: list[DetectionSample],
    image_size: int,
    confidence: float,
    batch_size: int,
    workers: int,
    device: str,
) -> dict[str, list[YoloBox]]:
    model = YOLO(str(model_path.resolve()))
    results = model.predict(
        source=[str(sample.image_path.resolve()) for sample in samples],
        imgsz=image_size,
        conf=confidence,
        batch=batch_size,
        workers=workers,
        device=device,
        verbose=False,
    )
    predictions: dict[str, list[YoloBox]] = {}
    for sample, result in zip(samples, results):
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xywhn = result.boxes.xywhn.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            boxes = [
                YoloBox(int(class_id), float(x), float(y), float(width), float(height), float(score))
                for (x, y, width, height), class_id, score in zip(xywhn, classes, confidences)
            ]
        predictions[str(sample.image_path.resolve())] = boxes
    return predictions


def match_image(
    ground_truth: list[YoloBox],
    predictions: list[YoloBox],
    iou_threshold: float,
) -> tuple[list[tuple[YoloBox, YoloBox, float]], list[YoloBox], list[YoloBox]]:
    matches = []
    unmatched_predictions = set(range(len(predictions)))
    misses = []
    for target in ground_truth:
        best_index = None
        best_iou = 0.0
        for index in unmatched_predictions:
            prediction = predictions[index]
            if prediction.class_id != target.class_id:
                continue
            current_iou = box_iou(target, prediction)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            matches.append((target, predictions[best_index], best_iou))
            unmatched_predictions.remove(best_index)
        else:
            misses.append(target)
    false_positives = [predictions[index] for index in sorted(unmatched_predictions)]
    return matches, misses, false_positives


def build_diagnostics(
    samples: list[DetectionSample],
    ground_truth: dict[str, list[YoloBox]],
    predictions: dict[str, list[YoloBox]],
    class_names: list[str],
    iou_threshold: float,
) -> dict:
    gt_counts: Counter = Counter()
    match_counts: Counter = Counter()
    miss_counts: Counter = Counter()
    false_positive_counts: Counter = Counter()
    sample_rows = []
    size_counts: Counter = Counter()
    size_misses: Counter = Counter()

    for sample in samples:
        key = str(sample.image_path.resolve())
        targets = ground_truth.get(key, [])
        detected = predictions.get(key, [])
        matches, misses, false_positives = match_image(targets, detected, iou_threshold)
        for target in targets:
            class_name = class_names[target.class_id]
            gt_counts[(class_name, sample.sensor, sample.scene)] += 1
            size_counts[(class_name, size_bucket(target))] += 1
        for target, _prediction, _iou in matches:
            match_counts[(class_names[target.class_id], sample.sensor, sample.scene)] += 1
        for target in misses:
            class_name = class_names[target.class_id]
            miss_counts[(class_name, sample.sensor, sample.scene)] += 1
            size_misses[(class_name, size_bucket(target))] += 1
        for prediction in false_positives:
            if 0 <= prediction.class_id < len(class_names):
                class_name = class_names[prediction.class_id]
            else:
                class_name = f"class_{prediction.class_id}"
            false_positive_counts[(class_name, sample.sensor, sample.scene)] += 1
        sample_rows.append(
            {
                "image_path": key,
                "image_name": sample.image_name,
                "sensor": sample.sensor,
                "scene": sample.scene,
                "gt_count": len(targets),
                "matched_count": len(matches),
                "missed_count": len(misses),
                "false_positive_count": len(false_positives),
                "missed_classes": ",".join(class_names[target.class_id] for target in misses),
            }
        )

    class_summary = []
    for class_name in class_names:
        total_gt = sum(count for (name, _sensor, _scene), count in gt_counts.items() if name == class_name)
        total_matched = sum(
            count for (name, _sensor, _scene), count in match_counts.items() if name == class_name
        )
        total_missed = sum(
            count for (name, _sensor, _scene), count in miss_counts.items() if name == class_name
        )
        total_fp = sum(
            count
            for (name, _sensor, _scene), count in false_positive_counts.items()
            if name == class_name
        )
        class_summary.append(
            {
                "class": class_name,
                "gt": int(total_gt),
                "matched": int(total_matched),
                "missed": int(total_missed),
                "false_positive": int(total_fp),
                "recall_at_iou": total_matched / total_gt if total_gt else None,
            }
        )

    slice_summary = []
    slice_keys = sorted(set(gt_counts) | set(match_counts) | set(miss_counts) | set(false_positive_counts))
    for class_name, sensor, scene in slice_keys:
        gt = gt_counts[(class_name, sensor, scene)]
        matched = match_counts[(class_name, sensor, scene)]
        missed = miss_counts[(class_name, sensor, scene)]
        false_positive = false_positive_counts[(class_name, sensor, scene)]
        slice_summary.append(
            {
                "class": class_name,
                "sensor": sensor,
                "scene": scene,
                "gt": int(gt),
                "matched": int(matched),
                "missed": int(missed),
                "false_positive": int(false_positive),
                "recall_at_iou": matched / gt if gt else None,
            }
        )

    size_summary = []
    for class_name, bucket in sorted(set(size_counts) | set(size_misses)):
        gt = size_counts[(class_name, bucket)]
        missed = size_misses[(class_name, bucket)]
        size_summary.append(
            {
                "class": class_name,
                "size_bucket": bucket,
                "gt": int(gt),
                "missed": int(missed),
                "miss_rate": missed / gt if gt else None,
            }
        )

    top_missed_samples = sorted(
        sample_rows,
        key=lambda row: (row["missed_count"], row["false_positive_count"], row["gt_count"]),
        reverse=True,
    )[:20]

    return {
        "image_count": len(samples),
        "iou_threshold": iou_threshold,
        "class_summary": class_summary,
        "slice_summary": slice_summary,
        "size_summary": size_summary,
        "top_missed_samples": top_missed_samples,
        "sample_rows": sample_rows,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main() -> None:
    args = parse_args()
    if not args.index.is_file():
        raise FileNotFoundError(args.index)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    samples = [sample for sample in read_scene_index(args.index) if sample.split == args.split]
    output_dir = args.output or args.model.resolve().parents[1] / "error_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = read_ground_truth(samples, CLASS_NAMES)
    predictions = collect_predictions(
        args.model,
        samples,
        args.image_size,
        args.confidence,
        args.batch_size,
        args.workers,
        args.device,
    )
    report = build_diagnostics(samples, ground_truth, predictions, CLASS_NAMES, args.iou_threshold)
    report.update(
        {
            "model": str(args.model.resolve()),
            "index": str(args.index.resolve()),
            "split": args.split,
            "confidence": args.confidence,
        }
    )

    (output_dir / "diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / "class_summary.csv",
        report["class_summary"],
        ["class", "gt", "matched", "missed", "false_positive", "recall_at_iou"],
    )
    write_csv(
        output_dir / "slice_summary.csv",
        report["slice_summary"],
        ["class", "sensor", "scene", "gt", "matched", "missed", "false_positive", "recall_at_iou"],
    )
    write_csv(
        output_dir / "size_summary.csv",
        report["size_summary"],
        ["class", "size_bucket", "gt", "missed", "miss_rate"],
    )
    write_csv(
        output_dir / "top_missed_samples.csv",
        report["top_missed_samples"],
        [
            "image_path",
            "image_name",
            "sensor",
            "scene",
            "gt_count",
            "matched_count",
            "missed_count",
            "false_positive_count",
            "missed_classes",
        ],
    )
    print(json.dumps({key: report[key] for key in ("image_count", "class_summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
