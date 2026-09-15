#!/usr/bin/env python3
"""Prepare ChallengeCup test-result files for the TZB evaluation requirements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OLD_CLASSES = ("soldier", "small_aircraft", "warship", "tank")
DEFAULT_NEW_CLASSES = ("patrol_boat", "armored_vehicle")


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {"payload_type": type(data).__name__}


def parse_classes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_file(path: Path, label: str) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([PROJECT_ROOT / path, SCRIPT_DIR / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("%s not found: %s" % (label, path))


def resolve_optional_file(path: Optional[Path], label: str) -> Optional[Path]:
    return resolve_file(path, label) if path is not None else None


def evaluate_yolo_checkpoint(data_yaml: Path, checkpoint: Path, device: str, image_size: int) -> Dict[str, Any]:
    import yaml

    from scene_recognition.detector_module.evaluate_yolo_same_protocol import (
        evaluate_checkpoint,
        read_class_names,
    )
    from scene_recognition.detector_module.resnet18_detector import YoloManifestDataset

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("test"):
        raise ValueError("data.yaml must contain a fixed test split: %s" % data_yaml)
    class_names = read_class_names(data_yaml)
    test_path = Path(str(config["test"]))
    if not test_path.is_absolute():
        test_path = data_yaml.parent / test_path
    dataset = YoloManifestDataset(test_path, len(class_names))
    metrics = evaluate_checkpoint(checkpoint, dataset, class_names, device, image_size)
    return {
        "data_yaml": str(data_yaml),
        "test_path": str(test_path),
        "test_images": len(dataset),
        "class_names": class_names,
        "checkpoint": str(checkpoint),
        "metrics": metrics,
    }


def aggregate_classes(metrics: Dict[str, Any], class_names: Sequence[str], metric_name: str) -> Dict[str, Any]:
    values: list[float] = []
    missing_support: list[str] = []
    per_class: Dict[str, Any] = {}
    for name in class_names:
        row = metrics.get("per_class", {}).get(name)
        if row is None or int(row.get("support", 0)) == 0:
            missing_support.append(name)
            continue
        value = float(row.get(metric_name, float("nan")))
        if not math.isnan(value):
            values.append(value)
            per_class[name] = {
                "support": int(row.get("support", 0)),
                metric_name: value,
            }
    return {
        "value": sum(values) / len(values) if values else None,
        "classes": list(class_names),
        "evaluated_classes": list(per_class),
        "missing_support": missing_support,
        "per_class": per_class,
    }


def retention(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or before <= 0:
        return None
    return after / before


def fps_from_npu_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {"available": False, "reason": "npu summary not provided"}
    images = int(summary.get("images", 0) or 0)
    if "avg_detector_ms" in summary and isinstance(summary["avg_detector_ms"], (int, float)):
        avg_ms = float(summary["avg_detector_ms"])
        return {
            "available": avg_ms > 0,
            "mode": summary.get("mode", "single"),
            "images": images,
            "avg_ms": round(avg_ms, 6),
            "fps": round(1000.0 / avg_ms, 6) if avg_ms > 0 else None,
        }
    branch_ms = summary.get("avg_detector_ms")
    route_counts = summary.get("route_counts")
    if isinstance(branch_ms, dict) and isinstance(route_counts, dict) and images > 0:
        detector_total = 0.0
        for route, count in route_counts.items():
            detector_total += int(count) * float(branch_ms.get(route, 0.0))
        avg_scene_ms = float(summary.get("avg_scene_ms", 0.0) or 0.0)
        avg_ms = avg_scene_ms + detector_total / images
        return {
            "available": avg_ms > 0,
            "mode": "routed",
            "images": images,
            "avg_scene_ms": round(avg_scene_ms, 6),
            "avg_detector_ms": branch_ms,
            "route_counts": route_counts,
            "avg_ms": round(avg_ms, 6),
            "fps": round(1000.0 / avg_ms, 6) if avg_ms > 0 else None,
        }
    return {"available": False, "reason": "unrecognized npu summary format", "summary_keys": sorted(summary.keys())}


def write_metrics_csv(path: Path, report: Dict[str, Any]) -> None:
    rows = []
    score = report["score_summary"]
    for key in (
        "base_before_map50",
        "base_after_map50",
        "krr_map50",
        "new_map50",
        "base_before_map50_95",
        "base_after_map50_95",
        "krr_map50_95",
        "new_map50_95",
        "fps",
    ):
        item = score.get(key, {})
        rows.append(
            {
                "metric": key,
                "value": item.get("value"),
                "target": item.get("target"),
                "passed": item.get("passed"),
                "note": item.get("note", ""),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "target", "passed", "note"])
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, report: Dict[str, Any]) -> None:
    score = report["score_summary"]
    lines = [
        "# TZB Test Result Summary",
        "",
        "## Required Metrics",
        "",
        "| Metric | Value | Target | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for key in ("base_after_map50", "krr_map50", "new_map50", "fps"):
        item = score.get(key, {})
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                key,
                "" if item.get("value") is None else item.get("value"),
                "" if item.get("target") is None else item.get("target"),
                item.get("passed"),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tzb_metrics.json`: full metric payload.",
            "- `tzb_metrics.csv`: compact spreadsheet summary.",
            "- `npu_fps_summary.json`: board-side FPS evidence from inference summary.",
            "- `agent_summary.json`: optional copy of Agent-style report summary, if provided.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_score_summary(
    base_before: Optional[Dict[str, Any]],
    base_after: Optional[Dict[str, Any]],
    increment_after: Optional[Dict[str, Any]],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    fps: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    def add(name: str, value: Optional[float], target: Optional[float], compare: str = ">=") -> None:
        passed = None
        if value is not None and target is not None:
            passed = value >= target if compare == ">=" else value <= target
        summary[name] = {
            "value": None if value is None else round(float(value), 8),
            "target": target,
            "passed": passed,
        }

    before_old_50 = after_old_50 = before_old_5095 = after_old_5095 = None
    new_50 = new_5095 = None
    if base_before and base_after:
        before_old_50 = aggregate_classes(base_before["metrics"], old_classes, "map50")["value"]
        after_old_50 = aggregate_classes(base_after["metrics"], old_classes, "map50")["value"]
        before_old_5095 = aggregate_classes(base_before["metrics"], old_classes, "map50_95")["value"]
        after_old_5095 = aggregate_classes(base_after["metrics"], old_classes, "map50_95")["value"]
    if increment_after:
        new_50 = aggregate_classes(increment_after["metrics"], new_classes, "map50")["value"]
        new_5095 = aggregate_classes(increment_after["metrics"], new_classes, "map50_95")["value"]

    add("base_before_map50", before_old_50, None)
    add("base_after_map50", after_old_50, 0.80)
    add("krr_map50", retention(before_old_50, after_old_50), 0.95)
    add("new_map50", new_50, 0.60)
    add("base_before_map50_95", before_old_5095, None)
    add("base_after_map50_95", after_old_5095, None)
    add("krr_map50_95", retention(before_old_5095, after_old_5095), None)
    add("new_map50_95", new_5095, None)

    fps_value = fps.get("fps") if fps.get("available") else None
    add("fps", fps_value, 30.0)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TZB metric and submission summary files.")
    parser.add_argument("--base-data", type=Path, help="Base test data.yaml with a fixed test split.")
    parser.add_argument("--increment-data", type=Path, help="Increment test data.yaml with a fixed test split.")
    parser.add_argument("--before-model", type=Path, help="Increment-before .pt checkpoint.")
    parser.add_argument("--after-model", type=Path, help="Increment-after .pt checkpoint.")
    parser.add_argument("--npu-summary", type=Path, help="Board inference summary.json for FPS.")
    parser.add_argument("--agent-summary", type=Path, help="Optional agent_reports/agent_summary.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-classes", default=",".join(DEFAULT_OLD_CLASSES))
    parser.add_argument("--new-classes", default=",".join(DEFAULT_NEW_CLASSES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--skip-map", action="store_true", help="Only package FPS/Agent summaries, do not run torch evaluation.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_classes = parse_classes(args.old_classes)
    new_classes = parse_classes(args.new_classes)
    npu_summary = read_json(resolve_optional_file(args.npu_summary, "npu summary"))
    agent_summary = read_json(resolve_optional_file(args.agent_summary, "agent summary"))
    fps = fps_from_npu_summary(npu_summary)

    base_before = base_after = increment_after = None
    if not args.skip_map:
        missing = [
            name
            for name, value in {
                "--base-data": args.base_data,
                "--increment-data": args.increment_data,
                "--before-model": args.before_model,
                "--after-model": args.after_model,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError("missing required map-evaluation args: %s" % ", ".join(missing))
        base_data = resolve_file(args.base_data, "base data")
        increment_data = resolve_file(args.increment_data, "increment data")
        before_model = resolve_file(args.before_model, "before model")
        after_model = resolve_file(args.after_model, "after model")
        base_before = evaluate_yolo_checkpoint(base_data, before_model, args.device, args.image_size)
        base_after = evaluate_yolo_checkpoint(base_data, after_model, args.device, args.image_size)
        increment_after = evaluate_yolo_checkpoint(increment_data, after_model, args.device, args.image_size)

    report = {
        "protocol": "tzb_board_test_summary",
        "requirements": {
            "base_test": "Use before and after models to compute base mAP and KRR.",
            "increment_test": "Use after model to compute New-mAP.",
            "fps": "Measured on Ascend 310B hardware from NPU inference summary.",
        },
        "old_classes": old_classes,
        "new_classes": new_classes,
        "base_before": base_before,
        "base_after": base_after,
        "increment_after": increment_after,
        "npu_fps": fps,
        "agent_summary": agent_summary,
        "score_summary": build_score_summary(
            base_before,
            base_after,
            increment_after,
            old_classes,
            new_classes,
            fps,
        ),
    }

    metrics_json = output_dir / "tzb_metrics.json"
    metrics_csv = output_dir / "tzb_metrics.csv"
    fps_json = output_dir / "npu_fps_summary.json"
    readme = output_dir / "README.md"
    metrics_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fps_json.write_text(json.dumps(fps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(metrics_csv, report)
    write_readme(readme, report)

    if agent_summary:
        (output_dir / "agent_summary.json").write_text(
            json.dumps(agent_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("[INFO] TZB metrics : %s" % metrics_json, flush=True)
    print("[INFO] TZB CSV     : %s" % metrics_csv, flush=True)
    print("[INFO] FPS summary : %s" % fps_json, flush=True)
    print("[INFO] README      : %s" % readme, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("prepare_tzb_submission.py: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
