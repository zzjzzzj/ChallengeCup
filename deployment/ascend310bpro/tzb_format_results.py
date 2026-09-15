#!/usr/bin/env python3
"""Create the TXT result folders required by TZB attachment 1."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TARGET_MODULE = "目标检测识别模块"
SCENE_MODULE = "场景认知模块"
DECISION_MODULE = "任务决策模块"
RESULT_GROUPS = (
    ("base_before", "基础模型-基础测试集推理结果"),
    ("base_after", "增量模型-基础测试集推理结果"),
    ("increment_after", "增量模型-增量测试集推理结果"),
)


def resolve_file(path: Path, label: str) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            PROJECT_ROOT / path,
            SCRIPT_DIR / path,
            Path.cwd() / path,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("%s not found: %s" % (label, path))


def resolve_optional_file(path: Optional[Path], label: str) -> Optional[Path]:
    return resolve_file(path, label) if path is not None else None


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {"payload": data}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("%s line %d is not a JSON object." % (path, line_no))
            rows.append(data)
    return rows


def image_size_from_row(row: Dict[str, Any]) -> Tuple[float, float]:
    raw_size = row.get("image_size")
    if isinstance(raw_size, list) and len(raw_size) >= 2:
        width, height = float(raw_size[0]), float(raw_size[1])
        if width > 0 and height > 0:
            return width, height
    image_path = Path(str(row.get("image", "")))
    if image_path.is_file():
        with Image.open(image_path) as image:
            return float(image.size[0]), float(image.size[1])
    raise ValueError("Cannot determine image size for row: %s" % row.get("image"))


def detection_box(detection: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    raw_box = detection.get("box") or detection.get("bbox")
    if isinstance(raw_box, list) and len(raw_box) >= 4:
        x1, y1, x2, y2 = [float(item) for item in raw_box[:4]]
        return x1, y1, x2, y2
    raw_xywh = detection.get("bbox_xywh") or detection.get("xywh")
    if isinstance(raw_xywh, list) and len(raw_xywh) >= 4:
        x, y, width, height = [float(item) for item in raw_xywh[:4]]
        return x, y, x + width, y + height
    return None


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def detection_to_yolo_line(detection: Dict[str, Any], image_width: float, image_height: float) -> Optional[str]:
    box = detection_box(detection)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1 = clamp(x1, 0.0, image_width)
    x2 = clamp(x2, 0.0, image_width)
    y1 = clamp(y1, 0.0, image_height)
    y2 = clamp(y2, 0.0, image_height)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None
    class_id = int(detection.get("class_id", 0))
    score = clamp(float(detection.get("score", detection.get("confidence", 0.0))))
    x_center = (x1 + width / 2.0) / image_width
    y_center = (y1 + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return "%d %.6f %.6f %.6f %.6f %.6f" % (
        class_id,
        clamp(x_center),
        clamp(y_center),
        clamp(norm_width),
        clamp(norm_height),
        score,
    )


def stem_for_row(row: Dict[str, Any]) -> str:
    image = str(row.get("image", "")).strip()
    if image:
        return Path(image).stem
    fallback = str(row.get("id") or row.get("name") or "").strip()
    if fallback:
        return Path(fallback).stem
    raise ValueError("Prediction row has no image path or id.")


def write_detection_txt(path: Path, row: Dict[str, Any]) -> int:
    image_width, image_height = image_size_from_row(row)
    lines: List[str] = []
    detections = row.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if isinstance(detection, dict):
                line = detection_to_yolo_line(detection, image_width, image_height)
                if line is not None:
                    lines.append(line)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_scene_txt(path: Path, row: Dict[str, Any]) -> None:
    scene = row.get("scene") if isinstance(row.get("scene"), dict) else {}
    if scene:
        name = str(scene.get("name") or scene.get("scene_name") or "unknown")
        confidence = float(scene.get("confidence", 0.0) or 0.0)
        path.write_text("%s %.6f\n" % (name, confidence), encoding="utf-8")
    else:
        path.write_text("none 0.000000\n", encoding="utf-8")


def write_decision_txt(path: Path, row: Dict[str, Any]) -> None:
    detector = row.get("detector") if isinstance(row.get("detector"), dict) else {}
    route = str(detector.get("route") or "single")
    reason = str(detector.get("route_reason") or "single_detector")
    elapsed = detector.get("elapsed_ms", "")
    input_size = detector.get("input_size", "")
    path.write_text(
        "route %s\nreason %s\ninput_size %s\nelapsed_ms %s\n"
        % (route, reason, input_size, elapsed),
        encoding="utf-8",
    )


def write_group(output_root: Path, group_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    group_dir = output_root / group_name
    target_dir = group_dir / TARGET_MODULE
    scene_dir = group_dir / SCENE_MODULE
    decision_dir = group_dir / DECISION_MODULE
    for directory in (target_dir, scene_dir, decision_dir):
        directory.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    for row in rows:
        stem = stem_for_row(row)
        total_detections += write_detection_txt(target_dir / ("%s.txt" % stem), row)
        write_scene_txt(scene_dir / ("%s.txt" % stem), row)
        write_decision_txt(decision_dir / ("%s.txt" % stem), row)
    return {
        "folder": str(group_dir),
        "target_module": str(target_dir),
        "scene_module": str(scene_dir),
        "decision_module": str(decision_dir),
        "images": len(rows),
        "detections": total_detections,
    }


def fps_value(summary: Dict[str, Any]) -> Optional[float]:
    value = summary.get("fps")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def copy_optional(source: Optional[Path], target: Path) -> Optional[str]:
    if source is None or not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def write_readme(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# TZB Test Result Folder",
        "",
        "This folder follows attachment 1 of the test/submission requirements.",
        "",
        "Required prediction groups:",
        "",
    ]
    for key, group_name in RESULT_GROUPS:
        group = payload["groups"][key]
        lines.append("- `%s/%s`: %d image TXT files" % (group_name, TARGET_MODULE, group["images"]))
    lines.extend(
        [
            "",
            "Detection TXT format:",
            "",
            "```text",
            "class x_center y_center width height confidence",
            "```",
            "",
            "Coordinates are normalized YOLO xywh values.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format predictions into TZB required TXT folders.")
    parser.add_argument("--base-before-jsonl", type=Path, required=True)
    parser.add_argument("--base-after-jsonl", type=Path, required=True)
    parser.add_argument("--increment-after-jsonl", type=Path, required=True)
    parser.add_argument("--base-before-summary", type=Path)
    parser.add_argument("--base-after-summary", type=Path)
    parser.add_argument("--increment-after-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--team-id", default="")
    parser.add_argument("--strategy-label", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for _, group_name in RESULT_GROUPS:
        shutil.rmtree(output_root / group_name, ignore_errors=True)
    shutil.rmtree(output_root / "FPS指标证明材料", ignore_errors=True)

    jsonl_paths = {
        "base_before": resolve_file(args.base_before_jsonl, "base-before predictions"),
        "base_after": resolve_file(args.base_after_jsonl, "base-after predictions"),
        "increment_after": resolve_file(args.increment_after_jsonl, "increment-after predictions"),
    }
    rows = {key: load_jsonl(path) for key, path in jsonl_paths.items()}
    groups = {
        key: write_group(output_root, group_name, rows[key])
        for key, group_name in RESULT_GROUPS
    }
    summary_paths = {
        "base_before": resolve_optional_file(args.base_before_summary, "base-before summary"),
        "base_after": resolve_optional_file(args.base_after_summary, "base-after summary"),
        "increment_after": resolve_optional_file(args.increment_after_summary, "increment-after summary"),
    }
    summaries = {
        key: read_json(path)
        for key, path in summary_paths.items()
    }
    fps = fps_value(summaries["increment_after"])

    evidence_dir = output_root / "FPS指标证明材料"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "base_before_summary": copy_optional(summary_paths["base_before"], evidence_dir / "base_before_summary.json"),
        "base_after_summary": copy_optional(summary_paths["base_after"], evidence_dir / "base_after_summary.json"),
        "increment_after_summary": copy_optional(summary_paths["increment_after"], evidence_dir / "increment_after_summary.json"),
    }
    (evidence_dir / "fps_summary.txt").write_text(
        "strategy %s\nfps %s\n"
        % (args.strategy_label or "unspecified", "" if fps is None else "%.3f" % fps),
        encoding="utf-8",
    )

    payload = {
        "team_id": args.team_id,
        "strategy_label": args.strategy_label,
        "output_dir": str(output_root),
        "groups": groups,
        "source_predictions": {key: str(path) for key, path in jsonl_paths.items()},
        "fps": fps,
        "evidence": copied,
    }
    (output_root / "tzb_result_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_root / "README.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
