#!/usr/bin/env python3
"""Cascade inference for Ascend 310B.

Pipeline:
1. Run the six-class main OM model at 960x960.
2. Run a single-class expert OM model at 1024x832.
3. Fuse the expert class with the main detections and keep other main classes.

The script supports two common YOLO OM output forms:
- raw YOLO head output: [1, 4 + C, N] or [N, 4 + C]
- postprocessed output: [N, 6], default columns x1,y1,x2,y2,conf,class_id
"""

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from infer_yolov8_om import AscendOmModel, iter_images


DEFAULT_MAIN_CLASSES = [
    "soldier",
    "small_aircraft",
    "warship",
    "tank",
    "patrol_boat",
    "armored_vehicle",
]


def infer_onnx_nchw_size(model_path: Path, input_name: str) -> Optional[Tuple[int, int]]:
    if model_path.suffix.lower() != ".onnx" or not model_path.is_file():
        return None
    try:
        import onnx  # type: ignore
    except ImportError:
        return None
    model = onnx.load(str(model_path))
    graph_inputs = list(model.graph.input)
    selected = next((item for item in graph_inputs if item.name == input_name), None)
    selected = selected or (graph_inputs[0] if graph_inputs else None)
    if selected is None:
        return None
    dims = selected.type.tensor_type.shape.dim
    if len(dims) != 4:
        return None
    height = int(dims[2].dim_value) if dims[2].dim_value else 0
    width = int(dims[3].dim_value) if dims[3].dim_value else 0
    if height > 0 and width > 0:
        return width, height
    return None


def infer_size_from_name(model_path: Path) -> Optional[Tuple[int, int]]:
    stem = model_path.stem
    rectangular_matches = re.findall(r"(?<!\d)(\d{3,5})x(\d{3,5})(?!\d)", stem)
    if rectangular_matches:
        width, height = rectangular_matches[-1]
        return int(width), int(height)
    square_matches = re.findall(r"(?<!\d)([4-9]\d{2}|1\d{3})(?!\d)", stem)
    if square_matches:
        size = int(square_matches[-1])
        return size, size
    return None


def resolve_input_size(
    model_path: Path,
    explicit_width: Optional[int],
    explicit_height: Optional[int],
    default_width: int,
    default_height: int,
    input_name: str,
    label: str,
) -> Tuple[int, int]:
    if (explicit_width is None) ^ (explicit_height is None):
        raise SystemExit("Pass both --%s-width and --%s-height, or neither." % (label, label))
    if explicit_width is not None and explicit_height is not None:
        return explicit_width, explicit_height

    inferred = infer_onnx_nchw_size(model_path, input_name)
    inferred_source = "ONNX input"
    if inferred is None:
        inferred = infer_size_from_name(model_path)
        inferred_source = "model filename"
    if inferred is not None:
        print(
            "[INFO] %s input size from %s: width=%d height=%d"
            % (label, inferred_source, inferred[0], inferred[1]),
            flush=True,
        )
        return inferred

    print(
        "[INFO] %s input size default: width=%d height=%d"
        % (label, default_width, default_height),
        flush=True,
    )
    return default_width, default_height


def resolve_soc_version(cli_value: Optional[str]) -> str:
    soc_version = cli_value or os.environ.get("SOC_VERSION")
    if not soc_version:
        raise SystemExit(
            "SOC version is required when converting ONNX to OM. "
            "Pass --soc-version Ascend310B4 or export SOC_VERSION=Ascend310B4."
        )
    return soc_version


def om_path_for_onnx(
    onnx_path: Path,
    input_width: int,
    input_height: int,
    soc_version: str,
    om_cache_dir: Optional[Path],
) -> Path:
    output_dir = om_cache_dir if om_cache_dir is not None else onnx_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / ("%s_%dx%d_%s.om" % (onnx_path.stem, input_width, input_height, soc_version))


def convert_onnx_to_om(
    onnx_path: Path,
    om_path: Path,
    input_width: int,
    input_height: int,
    input_name: str,
    soc_version: str,
    atc_bin: str,
    precision_mode: Optional[str],
    force: bool,
) -> Path:
    if om_path.is_file() and not force:
        print("[INFO] Reuse OM: %s" % om_path, flush=True)
        return om_path

    output_prefix = om_path.with_suffix("")
    command = [
        atc_bin,
        "--model=%s" % onnx_path,
        "--framework=5",
        "--output=%s" % output_prefix,
        "--input_format=NCHW",
        "--input_shape=%s:1,3,%d,%d" % (input_name, input_height, input_width),
        "--soc_version=%s" % soc_version,
    ]
    if precision_mode:
        command.append("--precision_mode=%s" % precision_mode)

    print("[INFO] Convert ONNX to OM:", flush=True)
    print("[INFO] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not om_path.is_file():
        raise FileNotFoundError("ATC finished but OM was not found: %s" % om_path)
    print("[INFO] Created OM: %s" % om_path, flush=True)
    return om_path


def resolve_model_for_npu(
    model_path: Path,
    input_width: int,
    input_height: int,
    input_name: str,
    args: argparse.Namespace,
) -> Path:
    suffix = model_path.suffix.lower()
    if suffix == ".om":
        return model_path
    if suffix != ".onnx":
        raise SystemExit("Model must be .om or .onnx: %s" % model_path)
    if not model_path.is_file():
        raise FileNotFoundError("ONNX model not found: %s" % model_path)
    soc_version = resolve_soc_version(args.soc_version)
    om_path = om_path_for_onnx(
        model_path,
        input_width,
        input_height,
        soc_version,
        args.om_cache_dir,
    )
    return convert_onnx_to_om(
        onnx_path=model_path,
        om_path=om_path,
        input_width=input_width,
        input_height=input_height,
        input_name=input_name,
        soc_version=soc_version,
        atc_bin=args.atc_bin,
        precision_mode=args.precision_mode,
        force=args.force_convert,
    )


@dataclass
class TransformMeta:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_x: int
    pad_y: int


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]
    source: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        x1, y1, x2, y2 = self.box_xyxy
        return {
            "class_id": self.class_id,
            "name": self.name,
            "confidence": round(float(self.confidence), 6),
            "bbox_xyxy": [round(float(x1), 3), round(float(y1), 3), round(float(x2), 3), round(float(y2), 3)],
            "bbox_xywh": [
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2 - x1), 3),
                round(float(y2 - y1), 3),
            ],
            "source": self.source,
            "notes": self.notes,
        }


def read_class_names(path: Optional[Path], defaults: Sequence[str]) -> List[str]:
    if path is None:
        return list(defaults)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def letterbox_rect(image: Image.Image, input_width: int, input_height: int) -> Tuple[np.ndarray, TransformMeta]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(float(input_width) / float(width), float(input_height) / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = rgb.resize((resized_width, resized_height), Image.BILINEAR)
    canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(array), TransformMeta(
        original_width=width,
        original_height=height,
        input_width=input_width,
        input_height=input_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def map_box_to_original(
    box: Tuple[float, float, float, float],
    transform: TransformMeta,
    coords: str,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if coords == "letterbox":
        x1 = (x1 - transform.pad_x) / transform.scale
        y1 = (y1 - transform.pad_y) / transform.scale
        x2 = (x2 - transform.pad_x) / transform.scale
        y2 = (y2 - transform.pad_y) / transform.scale
    return (
        clip(x1, 0.0, float(transform.original_width)),
        clip(y1, 0.0, float(transform.original_height)),
        clip(x2, 0.0, float(transform.original_width)),
        clip(y2, 0.0, float(transform.original_height)),
    )


def box_iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def classwise_nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    output: List[Detection] = []
    grouped: Dict[str, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.name, []).append(detection)
    for group in grouped.values():
        remaining = sorted(group, key=lambda item: item.confidence, reverse=True)
        while remaining:
            current = remaining.pop(0)
            output.append(current)
            remaining = [
                item
                for item in remaining
                if box_iou(current.box_xyxy, item.box_xyxy) <= iou_threshold
            ]
    return sorted(output, key=lambda item: item.confidence, reverse=True)


def weighted_fusion(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    output: List[Detection] = []
    grouped: Dict[str, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.name, []).append(detection)
    for class_name, group in grouped.items():
        remaining = sorted(group, key=lambda item: item.confidence, reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            kept = []
            for item in remaining:
                if box_iou(seed.box_xyxy, item.box_xyxy) >= iou_threshold:
                    cluster.append(item)
                else:
                    kept.append(item)
            remaining = kept
            weight_sum = sum(max(item.confidence, 1e-6) for item in cluster)
            fused_box = tuple(
                sum(item.box_xyxy[index] * max(item.confidence, 1e-6) for item in cluster) / weight_sum
                for index in range(4)
            )
            best = max(cluster, key=lambda item: item.confidence)
            sources = sorted({item.source for item in cluster})
            notes = ["fused_from=%d" % len(cluster), "sources=%s" % ",".join(sources)]
            output.append(
                Detection(
                    class_id=best.class_id,
                    name=class_name,
                    confidence=max(item.confidence for item in cluster),
                    box_xyxy=fused_box,  # type: ignore[arg-type]
                    source="fused",
                    notes=notes,
                )
            )
    return sorted(output, key=lambda item: item.confidence, reverse=True)


def infer_output_mode(output: np.ndarray, class_count: int) -> str:
    squeezed = np.squeeze(output)
    if squeezed.ndim == 2 and (squeezed.shape[0] in {6, 7} or squeezed.shape[1] in {6, 7}):
        return "nms"
    if output.size % 6 == 0 and output.size <= 6000:
        return "nms"
    channel_count = 4 + class_count
    if output.size % channel_count == 0:
        return "raw"
    raise ValueError("Cannot infer output mode from shape %s and %d classes" % (output.shape, class_count))


def reshape_raw_output(output: np.ndarray, class_count: int, layout: str) -> np.ndarray:
    channel_count = 4 + class_count
    squeezed = np.squeeze(output)
    if squeezed.ndim == 1:
        if squeezed.size % channel_count != 0:
            raise ValueError("Raw output size %d is not divisible by %d" % (squeezed.size, channel_count))
        if layout == "channels-last":
            return squeezed.reshape(-1, channel_count)
        return squeezed.reshape(channel_count, -1).T
    if squeezed.ndim == 2:
        if squeezed.shape[0] == channel_count and layout != "channels-last":
            return squeezed.T
        if squeezed.shape[1] == channel_count:
            return squeezed
    raise ValueError("Unsupported raw output shape: %s" % (squeezed.shape,))


def decode_raw_yolo(
    output: np.ndarray,
    class_names: Sequence[str],
    transform: TransformMeta,
    confidence_threshold: float,
    iou_threshold: float,
    layout: str,
    coords: str,
    source: str,
    min_box_size: float,
    forced_name: Optional[str] = None,
    forced_class_id: Optional[int] = None,
) -> List[Detection]:
    rows = reshape_raw_output(output, len(class_names), layout)
    detections: List[Detection] = []
    for row in rows:
        scores = row[4 : 4 + len(class_names)]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < confidence_threshold:
            continue
        x_center, y_center, width, height = [float(value) for value in row[:4]]
        box = (
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        )
        mapped = map_box_to_original(box, transform, coords)
        if mapped[2] - mapped[0] < min_box_size or mapped[3] - mapped[1] < min_box_size:
            continue
        final_class_id = forced_class_id if forced_class_id is not None else class_id
        final_name = forced_name if forced_name is not None else str(class_names[class_id])
        detections.append(
            Detection(
                class_id=final_class_id,
                name=final_name,
                confidence=confidence,
                box_xyxy=mapped,
                source=source,
            )
        )
    return classwise_nms(detections, iou_threshold)


def reshape_nms_output(output: np.ndarray, columns: int) -> np.ndarray:
    squeezed = np.squeeze(output)
    if squeezed.ndim == 1:
        if squeezed.size % columns != 0:
            raise ValueError("NMS output size %d is not divisible by %d" % (squeezed.size, columns))
        return squeezed.reshape(-1, columns)
    if squeezed.ndim == 2:
        if squeezed.shape[1] == columns:
            return squeezed
        if squeezed.shape[0] == columns:
            return squeezed.T
    raise ValueError("Unsupported NMS output shape: %s" % (squeezed.shape,))


def parse_nms_row(
    row: np.ndarray,
    nms_format: str,
) -> Tuple[Tuple[float, float, float, float], float, int]:
    if nms_format == "xyxy-conf-class":
        x1, y1, x2, y2, confidence, class_id = row[:6]
    elif nms_format == "class-conf-xyxy":
        class_id, confidence, x1, y1, x2, y2 = row[:6]
    elif nms_format == "conf-class-xyxy":
        confidence, class_id, x1, y1, x2, y2 = row[:6]
    elif nms_format == "xywh-conf-class":
        x_center, y_center, width, height, confidence, class_id = row[:6]
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
    else:
        raise ValueError("Unsupported NMS format: %s" % nms_format)
    return (
        (float(x1), float(y1), float(x2), float(y2)),
        float(confidence),
        int(round(float(class_id))),
    )


def decode_nms_output(
    output: np.ndarray,
    class_names: Sequence[str],
    transform: TransformMeta,
    confidence_threshold: float,
    iou_threshold: float,
    nms_format: str,
    coords: str,
    source: str,
    min_box_size: float,
    forced_name: Optional[str] = None,
    forced_class_id: Optional[int] = None,
) -> List[Detection]:
    rows = reshape_nms_output(output, 6)
    detections: List[Detection] = []
    for row in rows:
        box, confidence, class_id = parse_nms_row(row, nms_format)
        if confidence < confidence_threshold:
            continue
        if not np.isfinite(row).all():
            continue
        if class_id < 0 or class_id >= len(class_names):
            if forced_name is None:
                continue
            class_id = 0
        mapped = map_box_to_original(box, transform, coords)
        if mapped[2] - mapped[0] < min_box_size or mapped[3] - mapped[1] < min_box_size:
            continue
        final_class_id = forced_class_id if forced_class_id is not None else class_id
        final_name = forced_name if forced_name is not None else str(class_names[class_id])
        detections.append(
            Detection(
                class_id=final_class_id,
                name=final_name,
                confidence=confidence,
                box_xyxy=mapped,
                source=source,
            )
        )
    return classwise_nms(detections, iou_threshold)


def decode_model_output(
    output: np.ndarray,
    class_names: Sequence[str],
    transform: TransformMeta,
    confidence_threshold: float,
    iou_threshold: float,
    output_mode: str,
    raw_layout: str,
    nms_format: str,
    coords: str,
    source: str,
    min_box_size: float,
    forced_name: Optional[str] = None,
    forced_class_id: Optional[int] = None,
) -> List[Detection]:
    mode = output_mode
    if mode == "auto":
        mode = infer_output_mode(output, len(class_names))
    if mode == "raw":
        return decode_raw_yolo(
            output,
            class_names,
            transform,
            confidence_threshold,
            iou_threshold,
            raw_layout,
            coords,
            source,
            min_box_size,
            forced_name,
            forced_class_id,
        )
    if mode == "nms":
        return decode_nms_output(
            output,
            class_names,
            transform,
            confidence_threshold,
            iou_threshold,
            nms_format,
            coords,
            source,
            min_box_size,
            forced_name,
            forced_class_id,
        )
    raise ValueError("Unsupported output mode: %s" % output_mode)


def run_loaded_model_on_images(
    model: AscendOmModel,
    model_path: Path,
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    input_width: int,
    input_height: int,
    confidence: float,
    iou: float,
    output_mode: str,
    raw_layout: str,
    nms_format: str,
    coords: str,
    output_dtype: str,
    source: str,
    device_id: int,
    min_box_size: float,
    forced_name: Optional[str] = None,
    forced_class_id: Optional[int] = None,
) -> Dict[Path, Dict[str, object]]:
    results: Dict[Path, Dict[str, object]] = {}
    dtype = np.dtype(output_dtype)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensor, transform = letterbox_rect(image, input_width, input_height)
        started = time.perf_counter()
        outputs = model.infer(tensor.astype(np.float32), dtype)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not outputs:
            raise RuntimeError("Model produced no outputs: %s" % model_path)
        detections = decode_model_output(
            outputs[0],
            class_names,
            transform,
            confidence,
            iou,
            output_mode,
            raw_layout,
            nms_format,
            coords,
            source,
            min_box_size,
            forced_name,
            forced_class_id,
        )
        results[image_path] = {
            "detections": detections,
            "elapsed_ms": elapsed_ms,
            "input_shape": [1, 3, input_height, input_width],
            "raw_output_shape": list(outputs[0].shape),
            "raw_output_count": int(outputs[0].size),
        }
    return results


def run_model_on_images(
    model_path: Path,
    image_paths: Sequence[Path],
    class_names: Sequence[str],
    input_width: int,
    input_height: int,
    confidence: float,
    iou: float,
    output_mode: str,
    raw_layout: str,
    nms_format: str,
    coords: str,
    output_dtype: str,
    source: str,
    device_id: int,
    min_box_size: float,
    forced_name: Optional[str] = None,
    forced_class_id: Optional[int] = None,
) -> Dict[Path, Dict[str, object]]:
    with AscendOmModel(model_path, device_id) as model:
        return run_loaded_model_on_images(
            model,
            model_path,
            image_paths,
            class_names,
            input_width,
            input_height,
            confidence,
            iou,
            output_mode,
            raw_layout,
            nms_format,
            coords,
            output_dtype,
            source,
            device_id,
            min_box_size,
            forced_name,
            forced_class_id,
        )


def should_run_expert(detections: Sequence[Detection], expert_class: str, strategy: str, threshold: float) -> bool:
    if strategy == "always":
        return True
    expert_detections = [item for item in detections if item.name == expert_class]
    if strategy == "missing":
        return not expert_detections
    if strategy == "low-confidence":
        return bool(expert_detections) and max(item.confidence for item in expert_detections) < threshold
    if strategy == "missing-or-low-confidence":
        return (not expert_detections) or max(item.confidence for item in expert_detections) < threshold
    raise ValueError("Unsupported expert strategy: %s" % strategy)


def filter_and_remap_expert_detections(
    detections: Sequence[Detection],
    expert_class: str,
    main_class_id: int,
) -> List[Detection]:
    output: List[Detection] = []
    for detection in detections:
        if detection.name != expert_class:
            continue
        output.append(
            Detection(
                class_id=main_class_id,
                name=expert_class,
                confidence=detection.confidence,
                box_xyxy=detection.box_xyxy,
                source=detection.source,
                notes=detection.notes,
            )
        )
    return output


def fuse_final_detections(
    main_detections: Sequence[Detection],
    expert_detections: Sequence[Detection],
    expert_class: str,
    policy: str,
    iou_threshold: float,
) -> List[Detection]:
    non_expert_main = [item for item in main_detections if item.name != expert_class]
    main_expert = [item for item in main_detections if item.name == expert_class]
    if policy == "replace":
        expert_final = list(expert_detections) if expert_detections else list(main_expert)
    elif policy == "append":
        expert_final = classwise_nms([*main_expert, *expert_detections], iou_threshold)
    elif policy == "fuse":
        expert_final = weighted_fusion([*main_expert, *expert_detections], iou_threshold)
    else:
        raise ValueError("Unsupported expert fusion policy: %s" % policy)
    return sorted([*non_expert_main, *expert_final], key=lambda item: item.confidence, reverse=True)


def color_for_class(name: str) -> Tuple[int, int, int]:
    palette = {
        "soldier": (255, 64, 64),
        "small_aircraft": (64, 128, 255),
        "warship": (64, 180, 120),
        "tank": (240, 180, 64),
        "patrol_boat": (180, 96, 255),
        "armored_vehicle": (64, 210, 210),
    }
    return palette.get(name, (255, 255, 255))


def save_annotated(image_path: Path, detections: Sequence[Detection], output_path: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        color = color_for_class(detection.name)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = "%s %.2f" % (detection.name, detection.confidence)
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_path)


def relative_output_path(image_path: Path, input_path: Path, output_dir: Path) -> Path:
    if input_path.is_dir():
        try:
            relative = image_path.relative_to(input_path)
        except ValueError:
            relative = Path(image_path.name)
    else:
        relative = Path(image_path.name)
    return output_dir / relative.with_suffix(".jpg")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Six-class 960 + single-class 1024x832 NPU cascade inference.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--main-model", type=Path, required=True, help="Six-class 960 .om or .onnx model.")
    parser.add_argument("--expert-model", type=Path, required=True, help="Single-class expert .om or .onnx model.")
    parser.add_argument("--output-dir", type=Path, default=Path("cascade_outputs"))
    parser.add_argument("--summary", type=Path, help="Summary JSON path. Defaults to OUTPUT_DIR/summary.json.")
    parser.add_argument("--jsonl", type=Path, help="JSONL path. Defaults to OUTPUT_DIR/predictions.jsonl.")
    parser.add_argument("--main-classes", type=Path, help="Six-class names file. Defaults to the project class order.")
    parser.add_argument("--expert-classes", type=Path, help="Expert class names file. Defaults to the single expert class.")
    parser.add_argument("--expert-class", default="soldier")
    parser.add_argument("--main-width", type=int, help="Main model input width. Defaults to ONNX/filename inference, then 960.")
    parser.add_argument("--main-height", type=int, help="Main model input height. Defaults to ONNX/filename inference, then 960.")
    parser.add_argument("--expert-width", type=int, help="Expert model input width. Defaults to ONNX/filename inference, then 1024.")
    parser.add_argument("--expert-height", type=int, help="Expert model input height. Defaults to ONNX/filename inference, then 832.")
    parser.add_argument("--main-input-name", default="images")
    parser.add_argument("--expert-input-name", default="images")
    parser.add_argument("--soc-version", help="ATC soc_version, for example Ascend310B4. Defaults to SOC_VERSION env.")
    parser.add_argument("--atc-bin", default="atc")
    parser.add_argument("--precision-mode", default=None, help="Optional ATC precision_mode, for example allow_fp32_to_fp16.")
    parser.add_argument("--om-cache-dir", type=Path, help="Directory for auto-converted OM files. Defaults to ONNX directory.")
    parser.add_argument("--force-convert", action="store_true", help="Re-run ATC even when the cached OM already exists.")
    parser.add_argument("--main-conf", type=float, default=0.25)
    parser.add_argument("--expert-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--min-box-size", type=float, default=1.0, help="Drop decoded boxes narrower or shorter than this many original-image pixels.")
    parser.add_argument("--expert-trigger-conf", type=float, default=0.45)
    parser.add_argument(
        "--expert-strategy",
        default="always",
        choices=["always", "missing", "low-confidence", "missing-or-low-confidence"],
    )
    parser.add_argument("--fusion-policy", default="fuse", choices=["fuse", "replace", "append"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--main-output-mode", default="auto", choices=["auto", "raw", "nms"])
    parser.add_argument("--expert-output-mode", default="auto", choices=["auto", "raw", "nms"])
    parser.add_argument("--main-output-dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--expert-output-dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--raw-layout", default="channels-first", choices=["channels-first", "channels-last"])
    parser.add_argument(
        "--nms-format",
        default="xyxy-conf-class",
        choices=["xyxy-conf-class", "class-conf-xyxy", "conf-class-xyxy", "xywh-conf-class"],
    )
    parser.add_argument(
        "--main-nms-format",
        choices=["xyxy-conf-class", "class-conf-xyxy", "conf-class-xyxy", "xywh-conf-class"],
        help="Override --nms-format for the main model.",
    )
    parser.add_argument(
        "--expert-nms-format",
        choices=["xyxy-conf-class", "class-conf-xyxy", "conf-class-xyxy", "xywh-conf-class"],
        help="Override --nms-format for the expert model.",
    )
    parser.add_argument("--coords", default="letterbox", choices=["letterbox", "original"])
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary_path = args.summary or (args.output_dir / "summary.json")
    jsonl_path = args.jsonl or (args.output_dir / "predictions.jsonl")
    image_paths = iter_images(args.input)
    main_nms_format = args.main_nms_format or args.nms_format
    expert_nms_format = args.expert_nms_format or args.nms_format
    main_class_names = read_class_names(args.main_classes, DEFAULT_MAIN_CLASSES)
    if args.expert_class not in main_class_names:
        raise SystemExit("Expert class '%s' is not in main class names: %s" % (args.expert_class, main_class_names))
    expert_class_names = read_class_names(args.expert_classes, [args.expert_class])
    if args.expert_class not in expert_class_names:
        raise SystemExit("Expert class '%s' is not in expert class names: %s" % (args.expert_class, expert_class_names))
    expert_class_id = main_class_names.index(args.expert_class)
    main_width, main_height = resolve_input_size(
        args.main_model,
        args.main_width,
        args.main_height,
        960,
        960,
        args.main_input_name,
        "main",
    )
    expert_width, expert_height = resolve_input_size(
        args.expert_model,
        args.expert_width,
        args.expert_height,
        1024,
        832,
        args.expert_input_name,
        "expert",
    )
    main_model = resolve_model_for_npu(
        args.main_model,
        main_width,
        main_height,
        args.main_input_name,
        args,
    )
    expert_model = resolve_model_for_npu(
        args.expert_model,
        expert_width,
        expert_height,
        args.expert_input_name,
        args,
    )

    print("[INFO] Images: %d" % len(image_paths), flush=True)
    print("[INFO] Main model: %s" % main_model, flush=True)
    with AscendOmModel(main_model, args.device_id) as main_runner:
        main_results = run_loaded_model_on_images(
            main_runner,
            main_model,
            image_paths,
            main_class_names,
            main_width,
            main_height,
            args.main_conf,
            args.iou,
            args.main_output_mode,
            args.raw_layout,
            main_nms_format,
            args.coords,
            args.main_output_dtype,
            "main_6class_960",
            args.device_id,
            args.min_box_size,
        )

        expert_image_paths = [
            image_path
            for image_path in image_paths
            if should_run_expert(
                main_results[image_path]["detections"],  # type: ignore[arg-type]
                args.expert_class,
                args.expert_strategy,
                args.expert_trigger_conf,
            )
        ]
        print("[INFO] Expert images: %d" % len(expert_image_paths), flush=True)
        expert_results: Dict[Path, Dict[str, object]] = {}
        if expert_image_paths:
            with AscendOmModel(expert_model, args.device_id) as expert_runner:
                expert_results = run_loaded_model_on_images(
                    expert_runner,
                    expert_model,
                    expert_image_paths,
                    expert_class_names,
                    expert_width,
                    expert_height,
                    args.expert_conf,
                    args.iou,
                    args.expert_output_mode,
                    args.raw_layout,
                    expert_nms_format,
                    args.coords,
                    args.expert_output_dtype,
                    "expert_%s_%dx%d" % (args.expert_class, expert_width, expert_height),
                    args.device_id,
                    args.min_box_size,
                )
                for image_path, result in expert_results.items():
                    result["detections"] = filter_and_remap_expert_detections(
                        result["detections"],  # type: ignore[arg-type]
                        args.expert_class,
                        expert_class_id,
                    )

    rows = []
    for image_path in image_paths:
        main_detections = main_results[image_path]["detections"]  # type: ignore[assignment]
        expert_detections = expert_results.get(image_path, {}).get("detections", [])
        final_detections = fuse_final_detections(
            main_detections,  # type: ignore[arg-type]
            expert_detections,  # type: ignore[arg-type]
            args.expert_class,
            args.fusion_policy,
            args.iou,
        )
        if not args.no_save_images:
            save_annotated(
                image_path,
                final_detections,
                relative_output_path(image_path, args.input, args.output_dir / "images"),
            )
        row = {
            "image": str(image_path),
            "main": {
                "elapsed_ms": round(float(main_results[image_path]["elapsed_ms"]), 3),
                "detections": [item.to_dict() for item in main_detections],  # type: ignore[union-attr]
            },
            "expert": {
                "enabled": image_path in expert_results,
                "elapsed_ms": round(float(expert_results.get(image_path, {}).get("elapsed_ms", 0.0)), 3),
                "detections": [item.to_dict() for item in expert_detections],  # type: ignore[union-attr]
            },
            "final": {
                "detections": [item.to_dict() for item in final_detections],
            },
        }
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(jsonl_path, rows)
    print("[INFO] Summary: %s" % summary_path, flush=True)
    print("[INFO] JSONL: %s" % jsonl_path, flush=True)
    if not args.no_save_images:
        print("[INFO] Images: %s" % (args.output_dir / "images"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
