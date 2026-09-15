#!/usr/bin/env python3
"""Single six-class NPU inference for a compact best.onnx detector.

This entry point is for the common deployed detector output shape [N, 6]:

    x1, y1, x2, y2, confidence, class_id

It reuses the Ascend ACL runtime and decoding utilities from
``routed_infer_npu.py`` so ONNX-to-OM caching behaves the same as the routed
project.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from routed_infer_npu import (
    AscendOmModel,
    Detection,
    PROJECT_ROOT,
    SCRIPT_DIR,
    decode_hard_output,
    decode_easy_output,
    iter_images,
    prepare_detector_tensor,
    relative_output_path,
    resolve_model_for_npu,
    save_annotated_image,
    write_jsonl,
)


DEFAULT_MODEL_CANDIDATES = (
    SCRIPT_DIR / "models" / "best.onnx",
    SCRIPT_DIR.parent / "best.onnx",
    PROJECT_ROOT / "models" / "best.onnx",
)
DEFAULT_CLASSES = SCRIPT_DIR / "classes_6.txt"


def default_model_path() -> Path:
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_MODEL_CANDIDATES[0]


def _read_varint(data: bytes, index: int, limit: int) -> Tuple[int, int]:
    shift = 0
    value = 0
    while index < limit:
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, index
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")
    raise ValueError("truncated protobuf varint")


def _iter_proto_fields(data: bytes) -> Iterable[Tuple[int, int, int, int, Any]]:
    index = 0
    limit = len(data)
    while index < limit:
        key, index = _read_varint(data, index, limit)
        field_no = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            start = index
            value, index = _read_varint(data, index, limit)
            yield field_no, wire_type, start, index, value
        elif wire_type == 1:
            start = index
            index += 8
            if index > limit:
                raise ValueError("truncated protobuf fixed64")
            yield field_no, wire_type, start, index, data[start:index]
        elif wire_type == 2:
            length, index = _read_varint(data, index, limit)
            start = index
            index += length
            if index > limit:
                raise ValueError("truncated protobuf bytes")
            yield field_no, wire_type, start, index, data[start:index]
        elif wire_type == 5:
            start = index
            index += 4
            if index > limit:
                raise ValueError("truncated protobuf fixed32")
            yield field_no, wire_type, start, index, data[start:index]
        else:
            raise ValueError("unsupported protobuf wire type: %d" % wire_type)


def _first_message(data: bytes, field_no: int) -> Optional[bytes]:
    for item_field, wire_type, _start, _end, value in _iter_proto_fields(data):
        if item_field == field_no and wire_type == 2:
            return bytes(value)
    return None


def _string_field(data: bytes, field_no: int) -> Optional[str]:
    for item_field, wire_type, _start, _end, value in _iter_proto_fields(data):
        if item_field == field_no and wire_type == 2:
            try:
                return bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                return None
    return None


def _int_field(data: bytes, field_no: int) -> Optional[int]:
    for item_field, wire_type, _start, _end, value in _iter_proto_fields(data):
        if item_field == field_no and wire_type == 0:
            return int(value)
    return None


def _value_info_name_and_dims(value_info: bytes) -> Tuple[Optional[str], List[int]]:
    name = _string_field(value_info, 1)
    type_proto = _first_message(value_info, 2)
    if type_proto is None:
        return name, []
    tensor_type = _first_message(type_proto, 1)
    if tensor_type is None:
        return name, []
    shape = _first_message(tensor_type, 2)
    if shape is None:
        return name, []
    dims: List[int] = []
    for field_no, wire_type, _start, _end, value in _iter_proto_fields(shape):
        if field_no != 1 or wire_type != 2:
            continue
        dim_value = _int_field(bytes(value), 1)
        if dim_value is None or dim_value <= 0:
            return name, []
        dims.append(dim_value)
    return name, dims


def infer_static_onnx_nchw_size(model_path: Path, input_name: str) -> Optional[Tuple[int, int, str]]:
    if model_path.suffix.lower() != ".onnx" or not model_path.is_file():
        return None
    try:
        graph = _first_message(model_path.read_bytes(), 7)
    except (OSError, ValueError):
        return None
    if graph is None:
        return None

    candidates: List[Tuple[str, List[int]]] = []
    try:
        for field_no, wire_type, _start, _end, value in _iter_proto_fields(graph):
            if field_no != 11 or wire_type != 2:
                continue
            name, dims = _value_info_name_and_dims(bytes(value))
            if name and len(dims) == 4 and dims[2] > 0 and dims[3] > 0:
                candidates.append((name, dims))
    except ValueError:
        return None

    selected = next((item for item in candidates if item[0] == input_name), None)
    selected = selected or next((item for item in candidates if item[1][1] in (1, 3)), None)
    if selected is None:
        return None
    name, dims = selected
    return int(dims[3]), int(dims[2]), name


def infer_size_from_name(model_path: Path) -> Optional[Tuple[int, int]]:
    stem = model_path.stem
    rectangular_matches = re.findall(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", stem)
    if rectangular_matches:
        width, height = rectangular_matches[-1]
        return int(width), int(height)
    square_matches = re.findall(r"(?<!\d)([2-9]\d{2}|1\d{3})(?!\d)", stem)
    if square_matches:
        size = int(square_matches[-1])
        return size, size
    return None


def resolve_input_size(
    model_path: Path,
    explicit_width: Optional[int],
    explicit_height: Optional[int],
    input_name: str,
) -> Tuple[int, int]:
    if (explicit_width is None) ^ (explicit_height is None):
        raise ValueError("Pass both --width and --height, or neither.")
    if explicit_width is not None and explicit_height is not None:
        if explicit_width <= 0 or explicit_height <= 0:
            raise ValueError("--width/--height must be positive.")
        print("[INFO] single input size from CLI: width=%d height=%d" % (explicit_width, explicit_height), flush=True)
        return explicit_width, explicit_height

    inferred = infer_static_onnx_nchw_size(model_path, input_name)
    if inferred is not None:
        width, height, resolved_input_name = inferred
        print(
            "[INFO] single input size from ONNX input %s: width=%d height=%d"
            % (resolved_input_name, width, height),
            flush=True,
        )
        return width, height

    filename_size = infer_size_from_name(model_path)
    if filename_size is not None:
        width, height = filename_size
        print("[INFO] single input size from model filename: width=%d height=%d" % (width, height), flush=True)
        return width, height

    print("[INFO] single input size default: width=960 height=960", flush=True)
    return 960, 960


def read_class_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names:
        raise ValueError("classes file is empty: %s" % path)
    return names


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError("input not found: %s" % path)


def resolve_model_path(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            SCRIPT_DIR / path,
            PROJECT_ROOT / path,
            Path.cwd() / path,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "model not found: %s\nsearched:\n%s"
        % (path, "\n".join("  %s" % candidate for candidate in candidates))
    )


def resolve_classes_path(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates = [
            SCRIPT_DIR / path,
            PROJECT_ROOT / path,
            Path.cwd() / path,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "classes file not found: %s\nsearched:\n%s"
        % (path, "\n".join("  %s" % candidate for candidate in candidates))
    )


def run_single_image(
    model: AscendOmModel,
    model_path: Path,
    image_path: Path,
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        tensor, transform = prepare_detector_tensor(image, args.width, args.height)
        image_size = [image.size[0], image.size[1]]
    started = time.perf_counter()
    outputs = model.infer(tensor.astype(np.float32), np.dtype(args.output_dtype))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("model produced no outputs: %s" % model_path)

    if args.decode_mode == "raw":
        class_id_remap = {index: index for index in range(len(class_names))}
        detections = decode_hard_output(
            outputs[0],
            class_names,
            class_names,
            class_id_remap,
            transform,
            args.confidence,
            args.iou,
            args.raw_layout,
            args.coords,
            args.raw_box_format,
            args.raw_score_activation,
            args.min_box_size,
        )
    else:
        detections = decode_easy_output(
            outputs[0],
            class_names,
            transform,
            args.confidence,
            args.nms_format,
            args.coords,
            args.min_box_size,
            args.apply_nms,
            args.iou,
        )
    for detection in detections:
        detection.branch = "single"
    return {
        "image": str(image_path),
        "image_size": image_size,
        "detector": {
            "route": "single",
            "route_reason": "single_detector_model",
            "elapsed_ms": round(elapsed_ms, 3),
            "input_size": [args.width, args.height],
            "decode_mode": args.decode_mode,
            "output_count": int(outputs[0].size),
        },
        "detections": [detection.to_dict() for detection in detections],
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], model_path: Path, class_names: Sequence[str], args: argparse.Namespace) -> Dict[str, Any]:
    class_counts: Dict[str, int] = {}
    detector_times: List[float] = []
    total_detections = 0
    for row in rows:
        detector_times.append(float(row["detector"]["elapsed_ms"]))
        for detection in row["detections"]:
            total_detections += 1
            class_name = str(detection["class_name"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    avg_detector = sum(detector_times) / len(detector_times) if detector_times else 0.0
    fps = 1000.0 / avg_detector if avg_detector > 0 else None
    return {
        "mode": "single_6class",
        "images": len(rows),
        "total_detections": total_detections,
        "class_counts": dict(sorted(class_counts.items())),
        "avg_detector_ms": round(avg_detector, 3),
        "fps": round(fps, 3) if fps is not None else None,
        "fps_definition": "NPU detector inference FPS = 1000 / average detector elapsed_ms; excludes image decode, preprocessing, postprocessing, and file I/O.",
        "models": {"single": str(model_path)},
        "class_names": list(class_names),
        "confidence": float(args.confidence),
        "iou": float(args.iou),
        "input_size": [int(args.width), int(args.height)],
        "decode_mode": args.decode_mode,
        "raw_layout": args.raw_layout if args.decode_mode == "raw" else None,
        "raw_box_format": args.raw_box_format if args.decode_mode == "raw" else None,
        "raw_score_activation": args.raw_score_activation if args.decode_mode == "raw" else None,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single best.onnx six-class NPU inference for Ascend 310B.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="best.onnx or cached .om model.")
    parser.add_argument("--input", type=Path, help="Image file or directory. Not required with --convert-only.")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ascend310bpro_best"))
    parser.add_argument("--summary", type=Path, help="Summary JSON path. Default: OUTPUT_DIR/summary.json.")
    parser.add_argument("--jsonl", type=Path, help="Prediction JSONL path. Default: OUTPUT_DIR/predictions.jsonl.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--soc-version", help="ATC soc_version, for example Ascend310B4. Defaults to SOC_VERSION.")
    parser.add_argument("--atc-bin", default="atc")
    parser.add_argument("--precision-mode")
    parser.add_argument("--om-cache-dir", type=Path)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--width", type=int, help="Detector input width. Default: auto-detect from ONNX, then filename, then 960.")
    parser.add_argument("--height", type=int, help="Detector input height. Default: auto-detect from ONNX, then filename, then 960.")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--nms-format", default="xyxy-conf-class")
    parser.add_argument(
        "--decode-mode",
        choices=["nms", "raw"],
        default="nms",
        help="nms decodes compact [x1,y1,x2,y2,conf,class] output; raw decodes plain YOLO [4+nc,anchors] output.",
    )
    parser.add_argument("--raw-layout", choices=["channels-first", "channels-last"], default="channels-first")
    parser.add_argument("--raw-box-format", choices=["xywh", "xyxy"], default="xywh")
    parser.add_argument("--raw-score-activation", choices=["auto", "sigmoid", "raw"], default="auto")
    parser.add_argument("--coords", choices=["letterbox", "original"], default="letterbox")
    parser.add_argument("--apply-nms", action="store_true", help="Apply class-wise NMS to compact outputs.")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float32")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_source = resolve_model_path(args.model)
    args.width, args.height = resolve_input_size(model_source, args.width, args.height, args.input_name)
    model_path = resolve_model_for_npu(
        model_source,
        args.width,
        args.height,
        args.input_name,
        args.soc_version,
        args.atc_bin,
        args.precision_mode,
        args.force_convert,
        args.om_cache_dir,
    )
    if args.convert_only:
        print(json.dumps({"model": str(model_path)}, ensure_ascii=False, indent=2))
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --convert-only is used.")
    input_path = resolve_input_path(args.input)
    classes_path = resolve_classes_path(args.classes)
    class_names = read_class_names(classes_path)
    image_paths = iter_images(input_path)

    print("[INFO] Images: %d" % len(image_paths), flush=True)
    print("[INFO] Single model: %s" % model_path, flush=True)
    print("[INFO] Classes: %s" % classes_path, flush=True)

    output_rows: List[Dict[str, Any]] = []
    image_output_dir = args.output_dir / "images"
    with AscendOmModel(model_path, args.device_id) as model:
        for index, image_path in enumerate(image_paths, start=1):
            row = run_single_image(model, model_path, image_path, class_names, args)
            output_rows.append(row)
            if not args.no_save_images:
                detections = [
                    Detection(
                        box=tuple(item["box"]),
                        score=float(item["score"]),
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        branch=str(item.get("branch", "single")),
                    )
                    for item in row["detections"]
                ]
                save_annotated_image(
                    image_path,
                    relative_output_path(input_path, image_path, image_output_dir),
                    detections,
                    "none",
                    "single",
                )
            if index % 50 == 0 or index == len(image_paths):
                print("[INFO] Inferred %d/%d images" % (index, len(image_paths)), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.output_dir / "summary.json")
    jsonl_path = args.jsonl or (args.output_dir / "predictions.jsonl")
    summary = summarize_rows(output_rows, model_path, class_names, args)
    summary["predictions_jsonl"] = str(jsonl_path)
    if not args.no_save_images:
        summary["annotated_images"] = str(image_output_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(jsonl_path, output_rows)
    print("[INFO] Summary: %s" % summary_path, flush=True)
    print("[INFO] JSONL: %s" % jsonl_path, flush=True)
    if not args.no_save_images:
        print("[INFO] Images: %s" % image_output_dir, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("infer_best_6class_npu.py: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
