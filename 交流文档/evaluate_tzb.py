"""目标检测与增量学习指标评估

输入格式
--------
GT 标签：每行 ``class x_center y_center width height``
预测结果：每行 ``class x_center y_center width height confidence``

坐标均为归一化 YOLO 格式``xywh``。GT 与预测按同名 ``.txt`` 的文件名
（stem）配对；缺少预测文件视为整图漏检，只有预测而没有 GT 的文件视为
背景图上的检测。


--------
1. 评分使用 mAP@0.5。
2. KRR 严格使用同一基础测试集：
   ``新模型基础集旧类 mAP@0.5 / 旧模型基础集旧类 mAP@0.5``。

依赖：Python >= 3.9、NumPy。
直接运行本文件即可使用底部“固定路径配置区”的测试样例；也可用命令行覆盖路径。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10, dtype=np.float64)
RECALL_THRESHOLDS = np.linspace(0.00, 1.00, 101, dtype=np.float64)
MAX_DETECTIONS_PER_IMAGE = 300

BASE_MAP_THRESHOLDS = (
    (0.80, 30),
    (0.70, 25),
    (0.65, 20),
    (0.60, 15),
    (0.50, 10),
    (0.40, 5),
    (-math.inf, 0),
)
NEW_MAP_THRESHOLDS = ((0.60, 10), (0.50, 7), (0.40, 4), (-math.inf, 0))
KRR_THRESHOLDS = ((0.95, 10), (0.90, 7), (0.80, 4), (-math.inf, 0))
FPS_THRESHOLDS = ((30.0, 10), (20.0, 7), (10.0, 4), (-math.inf, 0))
GRADE_EPS = 1e-12


@dataclass(frozen=True)
class DetectionMetrics:
    """一个模型/测试集组合的逐类检测指标。"""

    ap: np.ndarray  # shape=(num_classes, 10)，无 GT 类为 nan
    n_gt: np.ndarray
    n_pred: np.ndarray
    num_images: int

    @property
    def ap50(self) -> np.ndarray:
        return self.ap[:, 0]

    @property
    def ap50_95(self) -> np.ndarray:
        valid_count = np.isfinite(self.ap).sum(axis=1)
        total = np.nansum(self.ap, axis=1)
        return np.divide(
            total,
            valid_count,
            out=np.full(self.ap.shape[0], np.nan, dtype=np.float64),
            where=valid_count > 0,
        )

    @property
    def map50(self) -> float:
        return _nanmean_or_nan(self.ap50)

    @property
    def map50_95(self) -> float:
        return _nanmean_or_nan(self.ap)


def _nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else float("nan")


def _class_id(value: float, path: Path, line_no: int) -> int:
    if not np.isfinite(value) or value < 0 or not float(value).is_integer():
        raise ValueError(f"{path} 第 {line_no} 行类别 id 必须是非负整数，实际为 {value!r}")
    return int(value)


def load_yolo_file(path: Path, with_confidence: bool) -> np.ndarray:
    """读取单个标签/预测文件，并做必要的数据校验。"""

    columns = 6 if with_confidence else 5
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != columns:
                expected = "class xc yc w h conf" if with_confidence else "class xc yc w h"
                raise ValueError(
                    f"{path} 第 {line_no} 行应恰好有 {columns} 列（{expected}），"
                    f"实际有 {len(parts)} 列：{line!r}"
                )
            try:
                row = [float(item) for item in parts]
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_no} 行包含非数值内容：{line!r}") from exc

            row[0] = float(_class_id(row[0], path, line_no))
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{path} 第 {line_no} 行包含 NaN 或无穷值")
            if row[3] <= 0 or row[4] <= 0:
                raise ValueError(f"{path} 第 {line_no} 行的 width/height 必须大于 0")
            if with_confidence and not 0.0 <= row[5] <= 1.0:
                raise ValueError(f"{path} 第 {line_no} 行置信度必须在 [0, 1] 内")
            rows.append(row)

    if not rows:
        return np.empty((0, columns), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def load_yolo_directory(directory: str | Path, with_confidence: bool) -> dict[str, np.ndarray]:
    """读取目录第一层的所有 ``.txt`` 文件，返回 ``{stem: ndarray}``。"""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.name,
    )
    if not files and not with_confidence:
        raise FileNotFoundError(f"GT 目录中没有 .txt 文件：{directory}")
    return {path.stem: load_yolo_file(path, with_confidence) for path in files}


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """将中心点格式 ``[xc, yc, w, h]`` 转为角点格式。"""

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    result = np.empty_like(boxes, dtype=np.float64)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def box_iou(labels_xyxy: np.ndarray, predictions_xyxy: np.ndarray) -> np.ndarray:
    """计算 ``N x M`` 两组框的两两 IoU；坐标可为归一化值或像素值。"""

    if len(labels_xyxy) == 0 or len(predictions_xyxy) == 0:
        return np.zeros((len(labels_xyxy), len(predictions_xyxy)), dtype=np.float64)
    top_left = np.maximum(labels_xyxy[:, None, :2], predictions_xyxy[None, :, :2])
    bottom_right = np.minimum(labels_xyxy[:, None, 2:], predictions_xyxy[None, :, 2:])
    intersection_wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    label_area = np.prod(np.clip(labels_xyxy[:, 2:] - labels_xyxy[:, :2], 0.0, None), axis=1)
    pred_area = np.prod(
        np.clip(predictions_xyxy[:, 2:] - predictions_xyxy[:, :2], 0.0, None), axis=1
    )
    union = label_area[:, None] + pred_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_predictions_yolo(
    labels: np.ndarray,
    predictions: np.ndarray,
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """
    每个 IoU 阈值独立匹配。候选对按 IoU 降序，先确保每个预测只使用一次，
    再确保每个 GT 只使用一次。类别不一致的框永不匹配。
    """

    num_predictions = len(predictions)
    correct = np.zeros((num_predictions, len(iou_thresholds)), dtype=bool)
    if len(labels) == 0 or num_predictions == 0:
        return correct

    label_boxes = xywh_to_xyxy(labels[:, 1:5])
    prediction_boxes = xywh_to_xyxy(predictions[:, 1:5])
    ious = box_iou(label_boxes, prediction_boxes)
    same_class = labels[:, 0:1].astype(np.int64) == predictions[None, :, 0].astype(np.int64)

    for threshold_index, threshold in enumerate(iou_thresholds):
        label_indices, prediction_indices = np.where((ious >= threshold) & same_class)
        if label_indices.size == 0:
            continue
        matches = np.column_stack(
            (label_indices, prediction_indices, ious[label_indices, prediction_indices])
        )
        if len(matches) > 1:
            matches = matches[np.argsort(-matches[:, 2], kind="mergesort")]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(np.int64), threshold_index] = True
    return correct


def coco_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:

    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if recall.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    sampled = np.zeros_like(RECALL_THRESHOLDS)
    valid = indices < recall.size
    sampled[valid] = envelope[indices[valid]]
    return float(sampled.mean())


def evaluate_detection(
    gt_by_image: dict[str, np.ndarray],
    predictions_by_image: dict[str, np.ndarray],
    num_classes: int,
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
) -> DetectionMetrics:
    """评估一个模型在一个测试集上的 AP50"""

    stems = sorted(set(gt_by_image) | set(predictions_by_image))
    if not stems:
        raise ValueError("GT 与预测均为空，无法评估")

    all_correct: list[np.ndarray] = []
    all_confidences: list[np.ndarray] = []
    all_prediction_classes: list[np.ndarray] = []
    all_target_classes: list[np.ndarray] = []

    empty_gt = np.empty((0, 5), dtype=np.float64)
    empty_pred = np.empty((0, 6), dtype=np.float64)
    for stem in stems:
        labels = gt_by_image.get(stem, empty_gt)
        predictions = predictions_by_image.get(stem, empty_pred)

        if len(predictions):
            order = np.argsort(-predictions[:, 5], kind="mergesort")
            predictions = predictions[order[:max_detections]]
            correct = match_predictions_yolo(labels, predictions)
            all_correct.append(correct)
            all_confidences.append(predictions[:, 5])
            all_prediction_classes.append(predictions[:, 0].astype(np.int64))
        if len(labels):
            all_target_classes.append(labels[:, 0].astype(np.int64))

    target_classes = (
        np.concatenate(all_target_classes) if all_target_classes else np.empty(0, dtype=np.int64)
    )
    n_gt = np.bincount(target_classes, minlength=num_classes)[:num_classes]

    if all_correct:
        correct = np.concatenate(all_correct, axis=0)
        confidence = np.concatenate(all_confidences)
        prediction_classes = np.concatenate(all_prediction_classes)
        global_order = np.argsort(-confidence, kind="mergesort")
        correct = correct[global_order]
        prediction_classes = prediction_classes[global_order]
    else:
        correct = np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
        prediction_classes = np.empty(0, dtype=np.int64)

    n_pred = np.bincount(prediction_classes, minlength=num_classes)[:num_classes]
    ap = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        if n_gt[class_id] == 0:
            continue  # 无 GT 类不进入 mAP 平均
        class_mask = prediction_classes == class_id
        class_correct = correct[class_mask]
        if len(class_correct) == 0:
            ap[class_id, :] = 0.0
            continue
        true_positive_cumsum = np.cumsum(class_correct, axis=0, dtype=np.float64)
        false_positive_cumsum = np.cumsum(~class_correct, axis=0, dtype=np.float64)
        recall = true_positive_cumsum / float(n_gt[class_id])
        precision = true_positive_cumsum / np.maximum(
            true_positive_cumsum + false_positive_cumsum, np.finfo(np.float64).eps
        )
        for threshold_index in range(len(IOU_THRESHOLDS)):
            ap[class_id, threshold_index] = coco_ap_101(
                recall[:, threshold_index], precision[:, threshold_index]
            )

    return DetectionMetrics(
        ap=ap,
        n_gt=n_gt.astype(np.int64),
        n_pred=n_pred.astype(np.int64),
        num_images=len(stems),
    )


def classes_present(data: dict[str, np.ndarray]) -> set[int]:
    result: set[int] = set()
    for rows in data.values():
        if len(rows):
            result.update(rows[:, 0].astype(np.int64).tolist())
    return result


def mean_for_classes(values: np.ndarray, class_ids: Iterable[int]) -> float:
    selected = np.asarray([values[class_id] for class_id in class_ids], dtype=np.float64)
    return _nanmean_or_nan(selected)


def grade(value: float, thresholds: Sequence[tuple[float, int]]) -> int | None:
    if not np.isfinite(value):
        return None
    for lower_bound, score in thresholds:
        if value >= lower_bound - GRADE_EPS:
            return score
    return 0


def parse_class_ids(text: str | None) -> list[int] | None:
    if text is None:
        return None
    if not text.strip():
        return []
    result = [int(part.strip()) for part in text.split(",")]
    if any(class_id < 0 for class_id in result):
        raise ValueError("类别 id 不能为负数")
    if len(set(result)) != len(result):
        raise ValueError(f"类别 id 不应重复：{result}")
    return result


def parse_names(value: str | Sequence[str] | None, num_classes: int) -> list[str]:
    if value is None:
        names: list[str] = []
    elif isinstance(value, str):
        possible_file = Path(value)
        if possible_file.is_file():
            names = [line.strip() for line in possible_file.read_text("utf-8-sig").splitlines() if line.strip()]
        else:
            names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value]
    if len(names) < num_classes:
        names.extend(f"class_{class_id}" for class_id in range(len(names), num_classes))
    return names[:num_classes]


def _format_value(value: float, digits: int = 4) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


def _metric_to_dict(metrics: DetectionMetrics, names: Sequence[str]) -> dict:
    per_class = []
    for class_id in range(len(names)):
        per_class.append(
            {
                "id": class_id,
                "name": names[class_id],
                "gt": int(metrics.n_gt[class_id]),
                "predictions": int(metrics.n_pred[class_id]),
                "ap50": None if not np.isfinite(metrics.ap50[class_id]) else float(metrics.ap50[class_id]),
            }
        )
    return {
        "images": metrics.num_images,
        "map50": metrics.map50,
        "per_class": per_class,
    }


def evaluate_all(
    base_gt_dir: str | Path,
    old_model_base_predictions: str | Path,
    new_model_base_predictions: str | Path,
    incremental_gt_dir: str | Path,
    new_model_incremental_predictions: str | Path,
    names: str | Sequence[str] | None = None,
    old_classes: Sequence[int] | None = None,
    new_classes: Sequence[int] | None = None,
    fps: float | None = None,
) -> dict:
    """计算四项评分指标并返回可序列化结果。"""

    base_gt = load_yolo_directory(base_gt_dir, with_confidence=False)
    old_base_pred = load_yolo_directory(old_model_base_predictions, with_confidence=True)
    new_base_pred = load_yolo_directory(new_model_base_predictions, with_confidence=True)
    incremental_gt = load_yolo_directory(incremental_gt_dir, with_confidence=False)
    new_incremental_pred = load_yolo_directory(
        new_model_incremental_predictions, with_confidence=True
    )

    all_data = (base_gt, old_base_pred, new_base_pred, incremental_gt, new_incremental_pred)
    all_classes = set().union(*(classes_present(data) for data in all_data))
    if not all_classes:
        raise ValueError("所有 GT 和预测文件中均没有目标")
    num_classes = max(all_classes) + 1
    class_names = parse_names(names, num_classes)

    base_gt_classes = classes_present(base_gt)
    incremental_gt_classes = classes_present(incremental_gt)
    old_class_ids = sorted(base_gt_classes if old_classes is None else set(old_classes))
    new_class_ids = sorted(
        incremental_gt_classes - base_gt_classes if new_classes is None else set(new_classes)
    )
    if not old_class_ids:
        raise ValueError("旧类别列表为空")
    if not new_class_ids:
        raise ValueError("新增类别列表为空；基础集可能未覆盖全部旧类，请显式设置 NEW_CLASSES")
    invalid = [class_id for class_id in old_class_ids + new_class_ids if not 0 <= class_id < num_classes]
    if invalid:
        raise ValueError(f"类别 id 超出 [0, {num_classes - 1}]：{invalid}")
    overlap = sorted(set(old_class_ids) & set(new_class_ids))
    if overlap:
        raise ValueError(f"旧类别与新增类别不能重叠：{overlap}")

    old_on_base = evaluate_detection(base_gt, old_base_pred, num_classes)
    new_on_base = evaluate_detection(base_gt, new_base_pred, num_classes)
    new_on_incremental = evaluate_detection(incremental_gt, new_incremental_pred, num_classes)

    base_map = old_on_base.map50
    new_map = mean_for_classes(new_on_incremental.ap50, new_class_ids)
    old_map_before = mean_for_classes(old_on_base.ap50, old_class_ids)
    old_map_after = mean_for_classes(new_on_base.ap50, old_class_ids)
    krr = (
        old_map_after / old_map_before
        if np.isfinite(old_map_before) and old_map_before > 0 and np.isfinite(old_map_after)
        else float("nan")
    )

    score_base = grade(base_map, BASE_MAP_THRESHOLDS)
    score_new = grade(new_map, NEW_MAP_THRESHOLDS)
    score_krr = grade(krr, KRR_THRESHOLDS)
    score_fps = None if fps is None else grade(float(fps), FPS_THRESHOLDS)
    score_values = [score_base, score_new, score_krr, score_fps]
    total_score = sum(score for score in score_values if score is not None)
    available_full_score = sum(
        full for score, full in zip(score_values, (30, 10, 10, 10)) if score is not None
    )

    print("=" * 78)
    print("目标检测与增量学习性能评估")
    print("=" * 78)

    absent_old = [class_id for class_id in old_class_ids if old_on_base.n_gt[class_id] == 0]
    absent_new = [class_id for class_id in new_class_ids if new_on_incremental.n_gt[class_id] == 0]
    if absent_old:
        print(f"\n注意：基础测试集无 GT 的旧类不参与 KRR 的 mAP 平均：{absent_old}")
    if absent_new:
        print(f"注意：增量测试集无 GT 的新类不参与 New-mAP 平均：{absent_new}")

    print("\n评价指标                            指标        得分")
    print("-" * 78)
    print(f"[1]基础 mAP@0.5                  = {_format_value(base_map)}  -> {score_base}/30")
    print(
        f"[2]New-mAP@0.5（类别 {new_class_ids}）     = {_format_value(new_map)}  "
        f"-> {score_new}/10"
    )
    print(
        "[3]KRR = 新模型基础集旧类 mAP@0.5 / 旧模型基础集旧类 mAP@0.5\n"
        f"       = {_format_value(old_map_after)} / {_format_value(old_map_before)}"
        f"        = {_format_value(krr, 6)}  -> {score_krr}/10"
    )
    if fps is None:
        print("[4]FPS                         = N/A  -> 未评分")
    else:
        print(f"[4]FPS                          = {fps:.2f}  -> {score_fps}/10")
    print(f"总分                             = {total_score}/{available_full_score}")

    return {
        "indicators": {
            "base_map50": base_map,
            "new_map50": new_map,
            "old_map50_before": old_map_before,
            "old_map50_after": old_map_after,
            "krr": krr,
            "fps": fps,
        },
        "scores": {
            "base_detection": score_base,
            "new_class_learning": score_new,
            "knowledge_retention": score_krr,
            "fps": score_fps,
            "total": total_score,
            "available_full_score": available_full_score,
        },
    }


def build_argument_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gt", default=str(defaults["base_gt"]), help="基础测试集 GT 目录")
    parser.add_argument(
        "--old-base-pred", default=str(defaults["old_base_pred"]), help="旧模型在基础测试集上的预测目录"
    )
    parser.add_argument(
        "--new-base-pred", default=str(defaults["new_base_pred"]), help="新模型在基础测试集上的预测目录"
    )
    parser.add_argument(
        "--incremental-gt", default=str(defaults["incremental_gt"]), help="增量测试集 GT 目录"
    )
    parser.add_argument(
        "--new-incremental-pred",
        default=str(defaults["new_incremental_pred"]),
        help="新模型在增量测试集上的预测目录",
    )
    parser.add_argument("--names", default=",".join(defaults["names"]), help="逗号分隔的类别名，或 names.txt")
    parser.add_argument("--old-classes", default=",".join(map(str, defaults["old_classes"])))
    parser.add_argument("--new-classes", default=",".join(map(str, defaults["new_classes"])))
    parser.add_argument("--fps", type=float, default=defaults["fps"], help="昇腾 310B 实测 FPS")
    parser.add_argument(
        "--no-fps", dest="fps", action="store_const", const=None, help="跳过 FPS 指标评分"
    )
    return parser


def _sample_root() -> Path:
    absolute_sample = Path(r"C:\Users\hc\Desktop\测试样例")
    local_sample = Path(__file__).resolve().parent / "测试样例" / "测试样例"
    return absolute_sample if absolute_sample.exists() else local_sample


def fixed_path_defaults() -> dict:
    """固定路径配置区：可直接修改这里，也可使用命令行参数覆盖。"""

    root = _sample_root()
    return {
        # 基础测试集 GT 目录
        "base_gt": r'E:\tzb_2026\比赛测试\gt\base_test_r1',
        # 旧模型在基础测试集上的预测目录
        "old_base_pred": r'E:\tzb_2026\比赛测试\pred\lables_oldmodel_base',
        # 新模型在基础测试集上的预测目录
        "new_base_pred": r'E:\tzb_2026\比赛测试\pred\lables_newmodel_base',
        #  增量测试集 GT 目录
        "incremental_gt": r'E:\tzb_2026\比赛测试\gt\inc_test_r2',
        # 新模型在增量测试集上的预测目录
        "new_incremental_pred": r'E:\tzb_2026\比赛测试\pred\lables_newmodel_inc',
        # 类别命名
        "names": ["soldier", "small_aircraft", "warship", "tank", "patrol_boat", "armored_vehicle"],
        "old_classes": [0, 1, 2, 3],
        "new_classes": [4, 5],
        "fps": 0,
    }


def main(argv: Sequence[str] | None = None) -> dict:
    defaults = fixed_path_defaults()
    args = build_argument_parser(defaults).parse_args(argv)
    result = evaluate_all(
        base_gt_dir=args.base_gt,
        old_model_base_predictions=args.old_base_pred,
        new_model_base_predictions=args.new_base_pred,
        incremental_gt_dir=args.incremental_gt,
        new_model_incremental_predictions=args.new_incremental_pred,
        names=args.names,
        old_classes=parse_class_ids(args.old_classes),
        new_classes=parse_class_ids(args.new_classes),
        fps=args.fps,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
