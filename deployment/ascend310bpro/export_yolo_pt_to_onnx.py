#!/usr/bin/env python3
"""Export a YOLO .pt checkpoint to a static batch-1 ONNX file."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


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
    raise FileNotFoundError(
        "%s not found: %s\nsearched:\n%s"
        % (label, path, "\n".join("  %s" % candidate for candidate in candidates))
    )


def import_yolo() -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Exporting .pt to ONNX requires ultralytics and torch in this Python. "
            "Use the training/export environment, or provide an already exported .onnx model."
        ) from exc
    return YOLO


def export_model(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = resolve_file(args.model, "model")
    if model_path.suffix.lower() != ".pt":
        raise ValueError("expected a .pt model, got: %s" % model_path)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not args.force:
        return {
            "reused": True,
            "pytorch_model": str(model_path),
            "onnx_model": str(output_path),
            "image_size": int(args.image_size),
        }

    YOLO = import_yolo()
    model = YOLO(str(model_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=int(args.image_size),
            batch=1,
            dynamic=False,
            simplify=bool(args.simplify),
            opset=int(args.opset),
            half=bool(args.half),
            device=str(args.device),
        )
    ).resolve()
    if not exported_path.is_file():
        fallback = model_path.with_suffix(".onnx")
        if fallback.is_file():
            exported_path = fallback.resolve()
        else:
            raise FileNotFoundError("Ultralytics export finished but ONNX was not found: %s" % exported_path)
    if exported_path != output_path:
        shutil.copy2(exported_path, output_path)

    summary = {
        "reused": False,
        "pytorch_model": str(model_path),
        "ultralytics_export": str(exported_path),
        "onnx_model": str(output_path),
        "onnx_bytes": output_path.stat().st_size,
        "image_size": int(args.image_size),
        "batch": 1,
        "dynamic": False,
        "simplify": bool(args.simplify),
        "opset": int(args.opset),
        "half": bool(args.half),
        "device": str(args.device),
    }
    summary_path = output_path.with_name(output_path.stem + "_export_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO .pt to static batch-1 ONNX.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = export_model(args)
    except Exception as exc:
        print("export_yolo_pt_to_onnx.py: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
