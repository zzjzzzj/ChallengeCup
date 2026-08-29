#!/usr/bin/env python3
"""Route one image through a scene classifier and exactly one detector.

Runtime design:
- scene router: 224x224 classifier, classes air/forest/sea/urban
- easy detector: 640x640 six-class YOLOv10 end-to-end output [N, 6]
- hard detector: 960x960 three-class YOLOv8 raw output [1, 7, N]

The script is intentionally independent from the training project. It does not
import torch, torchvision, or ultralytics. It is compatible with Python 3.9.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


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
    box: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    branch: str

    def to_dict(self) -> Dict[str, Any]:
        x1, y1, x2, y2 = self.box
        return {
            "box": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "bbox_xywh": [
                round(x1, 3),
                round(y1, 3),
                round(x2 - x1, 3),
                round(y2 - y1, 3),
            ],
            "score": round(float(self.score), 6),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "branch": self.branch,
        }


class AclError(RuntimeError):
    pass


class AscendOmModel:
    """Small ACL wrapper for static OM models with one input and one or more outputs."""

    _runtime_initialized = False
    _runtime_device_id: Optional[int] = None
    _active_models = 0

    def __init__(self, model_path: Path, device_id: int) -> None:
        try:
            import acl  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import 'acl'. Source CANN set_env.sh and make sure the "
                "Ascend ACL Python package is available in this environment."
            ) from exc

        self.acl = acl
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffer = None
        self.input_data_buffer = None
        self.output_buffers: List[Tuple[int, int, object]] = []
        self.initialized = False
        self.runtime_acquired = False

        self._acquire_runtime(device_id)
        self.context = self._value(self.acl.rt.create_context(device_id), "acl.rt.create_context")
        self.model_id = self._value(
            self.acl.mdl.load_from_file(str(model_path)),
            "acl.mdl.load_from_file",
        )
        self.model_desc = self.acl.mdl.create_desc()
        self._check(self.acl.mdl.get_desc(self.model_desc, self.model_id), "acl.mdl.get_desc")

        input_count = int(self._value(self.acl.mdl.get_num_inputs(self.model_desc), "acl.mdl.get_num_inputs"))
        if input_count != 1:
            raise RuntimeError("This runtime expects exactly one model input, got %d." % input_count)
        self.input_size = int(
            self._value(
                self.acl.mdl.get_input_size_by_index(self.model_desc, 0),
                "acl.mdl.get_input_size_by_index",
            )
        )
        self.input_buffer = self._value(
            self.acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_HUGE_FIRST),
            "acl.rt.malloc(input)",
        )
        self.input_data_buffer = self._value(
            self.acl.create_data_buffer(self.input_buffer, self.input_size),
            "acl.create_data_buffer(input)",
        )
        self.input_dataset = self.acl.mdl.create_dataset()
        self._check(
            self.acl.mdl.add_dataset_buffer(self.input_dataset, self.input_data_buffer),
            "acl.mdl.add_dataset_buffer(input)",
        )

        self.output_dataset = self.acl.mdl.create_dataset()
        output_count = int(self._value(self.acl.mdl.get_num_outputs(self.model_desc), "acl.mdl.get_num_outputs"))
        for index in range(output_count):
            output_size = int(
                self._value(
                    self.acl.mdl.get_output_size_by_index(self.model_desc, index),
                    "acl.mdl.get_output_size_by_index",
                )
            )
            output_ptr = self._value(
                self.acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST),
                "acl.rt.malloc(output)",
            )
            output_data_buffer = self._value(
                self.acl.create_data_buffer(output_ptr, output_size),
                "acl.create_data_buffer(output)",
            )
            self._check(
                self.acl.mdl.add_dataset_buffer(self.output_dataset, output_data_buffer),
                "acl.mdl.add_dataset_buffer(output)",
            )
            self.output_buffers.append((output_ptr, output_size, output_data_buffer))
        self.initialized = True

    def _acquire_runtime(self, device_id: int) -> None:
        cls = type(self)
        if cls._runtime_initialized:
            if cls._runtime_device_id != device_id:
                raise RuntimeError(
                    "ACL runtime is already initialized on device %s, cannot use device %s."
                    % (cls._runtime_device_id, device_id)
                )
            cls._active_models += 1
            self.runtime_acquired = True
            return
        self._check(self.acl.init(), "acl.init")
        self._check(self.acl.rt.set_device(device_id), "acl.rt.set_device")
        cls._runtime_initialized = True
        cls._runtime_device_id = device_id
        cls._active_models = 1
        self.runtime_acquired = True

    def _release_runtime(self) -> None:
        cls = type(self)
        if not self.runtime_acquired:
            return
        cls._active_models = max(0, cls._active_models - 1)
        self.runtime_acquired = False
        if cls._active_models == 0:
            if cls._runtime_device_id is not None:
                self._check(self.acl.rt.reset_device(cls._runtime_device_id), "acl.rt.reset_device")
            self._check(self.acl.finalize(), "acl.finalize")
            cls._runtime_initialized = False
            cls._runtime_device_id = None

    def infer(self, tensor: np.ndarray, output_dtype: np.dtype) -> List[np.ndarray]:
        tensor = np.ascontiguousarray(tensor)
        payload = tensor.tobytes()
        if len(payload) != self.input_size:
            raise ValueError(
                "Input tensor byte size mismatch: got %d, model expects %d."
                % (len(payload), self.input_size)
            )

        host_input = self.acl.util.bytes_to_ptr(payload)
        self._check(
            self.acl.rt.memcpy(
                self.input_buffer,
                self.input_size,
                host_input,
                self.input_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "acl.rt.memcpy(input)",
        )
        self._check(
            self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
            "acl.mdl.execute",
        )

        outputs: List[np.ndarray] = []
        for output_ptr, output_size, _data_buffer in self.output_buffers:
            host_output = self._value(self.acl.rt.malloc_host(output_size), "acl.rt.malloc_host(output)")
            try:
                self._check(
                    self.acl.rt.memcpy(
                        host_output,
                        output_size,
                        output_ptr,
                        output_size,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "acl.rt.memcpy(output)",
                )
                output_bytes = self.acl.util.ptr_to_bytes(host_output, output_size)
                outputs.append(np.frombuffer(output_bytes, dtype=output_dtype).copy())
            finally:
                self._check(self.acl.rt.free_host(host_output), "acl.rt.free_host")
        return outputs

    def close(self) -> None:
        if not self.initialized:
            return
        for output_ptr, _output_size, data_buffer in self.output_buffers:
            self._check(self.acl.destroy_data_buffer(data_buffer), "acl.destroy_data_buffer(output)")
            self._check(self.acl.rt.free(output_ptr), "acl.rt.free(output)")
        self.output_buffers = []
        if self.output_dataset is not None:
            self._check(self.acl.mdl.destroy_dataset(self.output_dataset), "acl.mdl.destroy_dataset(output)")
        if self.input_data_buffer is not None:
            self._check(self.acl.destroy_data_buffer(self.input_data_buffer), "acl.destroy_data_buffer(input)")
        if self.input_dataset is not None:
            self._check(self.acl.mdl.destroy_dataset(self.input_dataset), "acl.mdl.destroy_dataset(input)")
        if self.input_buffer is not None:
            self._check(self.acl.rt.free(self.input_buffer), "acl.rt.free(input)")
        if self.model_desc is not None:
            self._check(self.acl.mdl.destroy_desc(self.model_desc), "acl.mdl.destroy_desc")
        if self.model_id is not None:
            self._check(self.acl.mdl.unload(self.model_id), "acl.mdl.unload")
        if self.context is not None:
            self._check(self.acl.rt.destroy_context(self.context), "acl.rt.destroy_context")
        self._release_runtime()
        self.initialized = False

    def __enter__(self) -> "AscendOmModel":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _value(self, result, action: str):
        if isinstance(result, tuple):
            ret = result[-1]
            self._check(ret, action)
            if len(result) == 2:
                return result[0]
            return result[:-1]
        return result

    def _check(self, result, action: str) -> None:
        ret = result[-1] if isinstance(result, tuple) else result
        if ret != ACL_SUCCESS:
            raise AclError("%s failed with ret=%s" % (action, ret))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object." % label)
    return value


def resolve_relative_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def config_path_value(config: Dict[str, Any], branch: str, base_dir: Path) -> Path:
    section = require_mapping(config.get(branch), branch)
    return resolve_relative_path(str(section["model"]), base_dir)


def get_input_size(section: Dict[str, Any], label: str) -> Tuple[int, int]:
    size = section.get("input_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("%s.input_size must be [width, height]." % label)
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("%s input size must be positive." % label)
    return width, height


def get_classes(section: Dict[str, Any], label: str) -> List[str]:
    classes = section.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("%s.classes must be a non-empty list." % label)
    return [str(item) for item in classes]


def image_search_root(path: Path) -> Path:
    if path.is_dir():
        images_dir = path / "images"
        if images_dir.is_dir():
            return images_dir.resolve()
    return path.resolve()


def iter_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path.resolve()]
    if path.is_dir():
        search_root = image_search_root(path)
        images = [
            item.resolve()
            for item in sorted(search_root.rglob("*"))
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            return images
    raise FileNotFoundError("No image file found at %s" % path)


def resolve_soc_version(cli_value: Optional[str]) -> str:
    soc_version = cli_value or os.environ.get("SOC_VERSION")
    if not soc_version:
        raise SystemExit(
            "SOC version is required for ONNX to OM conversion. "
            "Pass --soc-version Ascend310B4 or export SOC_VERSION=Ascend310B4."
        )
    return soc_version


def om_path_for_onnx(
    onnx_path: Path,
    width: int,
    height: int,
    soc_version: str,
    cache_dir: Optional[Path],
) -> Path:
    output_dir = cache_dir.resolve() if cache_dir is not None else onnx_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / ("%s_%dx%d_%s.om" % (onnx_path.stem, width, height, soc_version))


def convert_onnx_to_om(
    onnx_path: Path,
    om_path: Path,
    width: int,
    height: int,
    input_name: str,
    soc_version: str,
    atc_bin: str,
    precision_mode: Optional[str],
    force: bool,
) -> Path:
    if om_path.is_file() and not force:
        print("[INFO] Reuse OM: %s" % om_path, flush=True)
        return om_path
    if not onnx_path.is_file():
        raise FileNotFoundError("ONNX model not found: %s" % onnx_path)
    command = [
        atc_bin,
        "--model=%s" % onnx_path,
        "--framework=5",
        "--output=%s" % om_path.with_suffix(""),
        "--input_format=NCHW",
        "--input_shape=%s:1,3,%d,%d" % (input_name, height, width),
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
    width: int,
    height: int,
    input_name: str,
    soc_version: Optional[str],
    atc_bin: str,
    precision_mode: Optional[str],
    force: bool,
    cache_dir: Optional[Path],
) -> Path:
    suffix = model_path.suffix.lower()
    if suffix == ".om":
        if not model_path.is_file():
            raise FileNotFoundError("OM model not found: %s" % model_path)
        return model_path
    if suffix != ".onnx":
        raise ValueError("Model must be .onnx or .om: %s" % model_path)
    resolved_soc_version = resolve_soc_version(soc_version)
    om_path = om_path_for_onnx(model_path, width, height, resolved_soc_version, cache_dir)
    return convert_onnx_to_om(
        model_path,
        om_path,
        width,
        height,
        input_name,
        resolved_soc_version,
        atc_bin,
        precision_mode,
        force,
    )


def prepare_scene_tensor(image: Image.Image, width: int, height: int, mode: str) -> np.ndarray:
    rgb = image.convert("RGB")
    if mode == "resize":
        prepared = rgb.resize((width, height), Image.BILINEAR)
    elif mode == "letterbox":
        prepared = letterbox_image(rgb, width, height)[0]
    else:
        raise ValueError("Unsupported scene preprocess mode: %s" % mode)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def letterbox_image(image: Image.Image, width: int, height: int) -> Tuple[Image.Image, TransformMeta]:
    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    scale = min(float(width) / float(original_width), float(height) / float(original_height))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = rgb.resize((resized_width, resized_height), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    pad_x = (width - resized_width) // 2
    pad_y = (height - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, TransformMeta(
        original_width=original_width,
        original_height=original_height,
        input_width=width,
        input_height=height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def prepare_detector_tensor(image: Image.Image, width: int, height: int) -> Tuple[np.ndarray, TransformMeta]:
    canvas, meta = letterbox_image(image, width, height)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...]), meta


def softmax(values: np.ndarray) -> np.ndarray:
    stable = values.astype(np.float64) - float(np.max(values))
    exp_values = np.exp(stable)
    return (exp_values / np.sum(exp_values)).astype(np.float32)


def normalize_scene_scores(raw_scores: np.ndarray, score_mode: str) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    if score_mode == "softmax":
        return softmax(scores)
    if score_mode == "raw":
        return scores
    if score_mode != "auto":
        raise ValueError("Unsupported scene score mode: %s" % score_mode)
    finite = scores[np.isfinite(scores)]
    if finite.size == scores.size and np.min(scores) >= 0.0 and np.max(scores) <= 1.0:
        total = float(np.sum(scores))
        if 0.95 <= total <= 1.05:
            return scores
    return softmax(scores)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def normalize_detection_scores(scores: np.ndarray, activation: str) -> np.ndarray:
    if activation == "sigmoid":
        return sigmoid(scores)
    if activation == "raw":
        return scores.astype(np.float32)
    if activation != "auto":
        raise ValueError("Unsupported score activation: %s" % activation)
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        return sigmoid(scores)
    return scores.astype(np.float32)


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def restore_box(
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
    elif coords != "original":
        raise ValueError("Unsupported coordinate space: %s" % coords)
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
    grouped: Dict[int, List[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.class_id, []).append(detection)
    output: List[Detection] = []
    for group in grouped.values():
        pending = sorted(group, key=lambda item: item.score, reverse=True)
        while pending:
            best = pending.pop(0)
            output.append(best)
            pending = [item for item in pending if box_iou(best.box, item.box) <= iou_threshold]
    return sorted(output, key=lambda item: item.score, reverse=True)


def reshape_nms_output(output: np.ndarray, columns: int) -> np.ndarray:
    squeezed = np.squeeze(output)
    if squeezed.ndim == 1:
        if squeezed.size % columns != 0:
            raise ValueError("NMS output size %d is not divisible by %d." % (squeezed.size, columns))
        return squeezed.reshape(-1, columns)
    if squeezed.ndim == 2:
        if squeezed.shape[1] == columns:
            return squeezed
        if squeezed.shape[0] == columns:
            return squeezed.T
    raise ValueError("Unsupported NMS output shape: %s" % (squeezed.shape,))


def parse_nms_row(row: np.ndarray, nms_format: str) -> Tuple[Tuple[float, float, float, float], float, int]:
    if nms_format == "xyxy-conf-class":
        x1, y1, x2, y2, score, class_id = row[:6]
    elif nms_format == "class-conf-xyxy":
        class_id, score, x1, y1, x2, y2 = row[:6]
    elif nms_format == "conf-class-xyxy":
        score, class_id, x1, y1, x2, y2 = row[:6]
    elif nms_format == "xywh-conf-class":
        x_center, y_center, width, height, score, class_id = row[:6]
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
    else:
        raise ValueError("Unsupported NMS format: %s" % nms_format)
    return (
        (float(x1), float(y1), float(x2), float(y2)),
        float(score),
        int(round(float(class_id))),
    )


def decode_easy_output(
    output: np.ndarray,
    class_names: Sequence[str],
    transform: TransformMeta,
    confidence: float,
    nms_format: str,
    coords: str,
    min_box_size: float,
    apply_nms: bool,
    iou: float,
) -> List[Detection]:
    rows = reshape_nms_output(output, 6)
    detections: List[Detection] = []
    for row in rows:
        if not np.isfinite(row).all():
            continue
        raw_box, score, class_id = parse_nms_row(row, nms_format)
        if score < confidence or class_id < 0 or class_id >= len(class_names):
            continue
        box = restore_box(raw_box, transform, coords)
        if box[2] - box[0] < min_box_size or box[3] - box[1] < min_box_size:
            continue
        detections.append(
            Detection(
                box=box,
                score=score,
                class_id=class_id,
                class_name=str(class_names[class_id]),
                branch="easy",
            )
        )
    detections = sorted(detections, key=lambda item: item.score, reverse=True)
    return classwise_nms(detections, iou) if apply_nms else detections


def reshape_raw_output(output: np.ndarray, class_count: int, layout: str) -> np.ndarray:
    channel_count = 4 + class_count
    squeezed = np.squeeze(output)
    if squeezed.ndim == 1:
        if squeezed.size % channel_count != 0:
            raise ValueError("Raw output size %d is not divisible by %d." % (squeezed.size, channel_count))
        if layout == "channels-last":
            return squeezed.reshape(-1, channel_count)
        return squeezed.reshape(channel_count, -1).T
    if squeezed.ndim == 2:
        if squeezed.shape[0] == channel_count and layout != "channels-last":
            return squeezed.T
        if squeezed.shape[1] == channel_count:
            return squeezed
    raise ValueError("Unsupported raw output shape: %s" % (squeezed.shape,))


def raw_box_to_xyxy(values: Sequence[float], box_format: str) -> Tuple[float, float, float, float]:
    if box_format == "xywh":
        x_center, y_center, width, height = [float(item) for item in values[:4]]
        return (
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        )
    if box_format == "xyxy":
        x1, y1, x2, y2 = [float(item) for item in values[:4]]
        return x1, y1, x2, y2
    raise ValueError("Unsupported raw box format: %s" % box_format)


def decode_hard_output(
    output: np.ndarray,
    local_class_names: Sequence[str],
    global_class_names: Sequence[str],
    class_id_remap: Dict[int, int],
    transform: TransformMeta,
    confidence: float,
    iou: float,
    raw_layout: str,
    coords: str,
    box_format: str,
    score_activation: str,
    min_box_size: float,
) -> List[Detection]:
    rows = reshape_raw_output(output, len(local_class_names), raw_layout)
    detections: List[Detection] = []
    for row in rows:
        if not np.isfinite(row).all():
            continue
        scores = normalize_detection_scores(row[4 : 4 + len(local_class_names)], score_activation)
        local_class_id = int(np.argmax(scores))
        score = float(scores[local_class_id])
        if score < confidence:
            continue
        if local_class_id not in class_id_remap:
            continue
        global_class_id = int(class_id_remap[local_class_id])
        if global_class_id < 0 or global_class_id >= len(global_class_names):
            continue
        raw_box = raw_box_to_xyxy(row[:4], box_format)
        box = restore_box(raw_box, transform, coords)
        if box[2] - box[0] < min_box_size or box[3] - box[1] < min_box_size:
            continue
        detections.append(
            Detection(
                box=box,
                score=score,
                class_id=global_class_id,
                class_name=str(global_class_names[global_class_id]),
                branch="hard",
            )
        )
    return classwise_nms(detections, iou)


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    palette = {
        "soldier": (255, 64, 64),
        "small_aircraft": (64, 128, 255),
        "warship": (64, 180, 120),
        "tank": (240, 180, 64),
        "patrol_boat": (180, 96, 255),
        "armored_vehicle": (64, 210, 210),
    }
    return palette.get(class_name, (255, 255, 255))


def save_annotated_image(
    image_path: Path,
    output_path: Path,
    detections: Sequence[Detection],
    scene_name: str,
    route: str,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = color_for_class(detection.class_name)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = "%s %.2f" % (detection.class_name, detection.score)
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=color)
    draw.text((6, 6), "scene=%s route=%s" % (scene_name, route), fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def relative_output_path(input_root: Path, image_path: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        base = image_search_root(input_root)
        try:
            relative = image_path.relative_to(base)
        except ValueError:
            relative = Path(image_path.name)
    else:
        relative = Path(image_path.name)
    return output_dir / relative.with_suffix(".jpg")


def run_scene_router(
    model: AscendOmModel,
    image_path: Path,
    section: Dict[str, Any],
) -> Dict[str, Any]:
    width, height = get_input_size(section, "scene_router")
    class_names = get_classes(section, "scene_router")
    with Image.open(image_path) as image:
        tensor = prepare_scene_tensor(image, width, height, str(section.get("preprocess", "resize")))
    started = time.perf_counter()
    outputs = model.infer(tensor.astype(np.float32), np.dtype(str(section.get("output_dtype", "float32"))))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("Scene router produced no outputs.")
    scores = normalize_scene_scores(outputs[0], str(section.get("score_mode", "auto")))
    if scores.size < len(class_names):
        raise ValueError("Scene output has %d scores but %d classes are configured." % (scores.size, len(class_names)))
    scores = scores[: len(class_names)]
    scene_id = int(np.argmax(scores))
    return {
        "scene_id": scene_id,
        "scene_name": class_names[scene_id],
        "confidence": float(scores[scene_id]),
        "scores": [float(item) for item in scores],
        "elapsed_ms": elapsed_ms,
    }


def choose_route(scene_result: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, str]:
    route_confidence = float(config.get("route_confidence", 0.60))
    uncertain_route = str(config.get("uncertain_route", "hard"))
    scene_section = require_mapping(config["scene_router"], "scene_router")
    easy_scenes = {str(item) for item in scene_section.get("easy_scenes", ["air", "sea"])}
    if float(scene_result["confidence"]) < route_confidence:
        return uncertain_route, "scene_confidence_below_threshold"
    if str(scene_result["scene_name"]) in easy_scenes:
        return "easy", "confident_easy_scene"
    return "hard", "confident_hard_scene"


def run_easy_detector(
    model: AscendOmModel,
    image_path: Path,
    section: Dict[str, Any],
    min_box_size: float,
) -> Tuple[List[Detection], Dict[str, Any]]:
    width, height = get_input_size(section, "easy_branch")
    class_names = get_classes(section, "easy_branch")
    with Image.open(image_path) as image:
        tensor, transform = prepare_detector_tensor(image, width, height)
    started = time.perf_counter()
    outputs = model.infer(tensor.astype(np.float32), np.dtype(str(section.get("output_dtype", "float32"))))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("Easy detector produced no outputs.")
    detections = decode_easy_output(
        outputs[0],
        class_names,
        transform,
        float(section.get("confidence", 0.25)),
        str(section.get("nms_format", "xyxy-conf-class")),
        str(section.get("coords", "letterbox")),
        min_box_size,
        bool(section.get("apply_nms", False)),
        float(section.get("iou", 0.55)),
    )
    return detections, {
        "elapsed_ms": elapsed_ms,
        "input_size": [width, height],
        "output_count": int(outputs[0].size),
    }


def parse_remap(raw_mapping: Any) -> Dict[int, int]:
    mapping = require_mapping(raw_mapping, "hard_branch.class_id_remap")
    return {int(key): int(value) for key, value in mapping.items()}


def run_hard_detector(
    model: AscendOmModel,
    image_path: Path,
    section: Dict[str, Any],
    global_class_names: Sequence[str],
    min_box_size: float,
) -> Tuple[List[Detection], Dict[str, Any]]:
    width, height = get_input_size(section, "hard_branch")
    local_class_names = get_classes(section, "hard_branch")
    class_id_remap = parse_remap(section.get("class_id_remap", {"0": 0, "1": 3, "2": 5}))
    with Image.open(image_path) as image:
        tensor, transform = prepare_detector_tensor(image, width, height)
    started = time.perf_counter()
    outputs = model.infer(tensor.astype(np.float32), np.dtype(str(section.get("output_dtype", "float32"))))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("Hard detector produced no outputs.")
    detections = decode_hard_output(
        outputs[0],
        local_class_names,
        global_class_names,
        class_id_remap,
        transform,
        float(section.get("confidence", 0.25)),
        float(section.get("iou", 0.55)),
        str(section.get("raw_layout", "channels-first")),
        str(section.get("coords", "letterbox")),
        str(section.get("box_format", "xywh")),
        str(section.get("score_activation", "auto")),
        min_box_size,
    )
    return detections, {
        "elapsed_ms": elapsed_ms,
        "input_size": [width, height],
        "output_count": int(outputs[0].size),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], model_paths: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    route_counts: Dict[str, int] = {}
    scene_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    scene_times: List[float] = []
    detector_times: Dict[str, List[float]] = {"easy": [], "hard": []}
    total_detections = 0
    for row in rows:
        scene = row["scene"]
        detector = row["detector"]
        route_counts[detector["route"]] = route_counts.get(detector["route"], 0) + 1
        scene_counts[scene["name"]] = scene_counts.get(scene["name"], 0) + 1
        scene_times.append(float(scene["elapsed_ms"]))
        detector_times[detector["route"]].append(float(detector["elapsed_ms"]))
        for detection in row["detections"]:
            total_detections += 1
            class_name = str(detection["class_name"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    avg_scene = sum(scene_times) / len(scene_times) if scene_times else 0.0
    avg_detector = {
        key: (sum(values) / len(values) if values else 0.0)
        for key, values in detector_times.items()
    }
    return {
        "images": len(rows),
        "total_detections": total_detections,
        "route_counts": dict(sorted(route_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "avg_scene_ms": round(avg_scene, 3),
        "avg_detector_ms": {key: round(value, 3) for key, value in avg_detector.items()},
        "models": model_paths,
        "route_confidence": float(config.get("route_confidence", 0.60)),
        "uncertain_route": str(config.get("uncertain_route", "hard")),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.route_confidence is not None:
        config["route_confidence"] = args.route_confidence
    if args.scene_model is not None:
        require_mapping(config["scene_router"], "scene_router")["model"] = str(args.scene_model.resolve())
    if args.easy_model is not None:
        require_mapping(config["easy_branch"], "easy_branch")["model"] = str(args.easy_model.resolve())
    if args.hard_model is not None:
        require_mapping(config["hard_branch"], "hard_branch")["model"] = str(args.hard_model.resolve())
    if args.det_conf is not None:
        require_mapping(config["easy_branch"], "easy_branch")["confidence"] = args.det_conf
        require_mapping(config["hard_branch"], "hard_branch")["confidence"] = args.det_conf
    if args.easy_conf is not None:
        require_mapping(config["easy_branch"], "easy_branch")["confidence"] = args.easy_conf
    if args.hard_conf is not None:
        require_mapping(config["hard_branch"], "hard_branch")["confidence"] = args.hard_conf
    if args.hard_iou is not None:
        require_mapping(config["hard_branch"], "hard_branch")["iou"] = args.hard_iou
    if args.scene_score_mode is not None:
        require_mapping(config["scene_router"], "scene_router")["score_mode"] = args.scene_score_mode
    if args.hard_score_activation is not None:
        require_mapping(config["hard_branch"], "hard_branch")["score_activation"] = args.hard_score_activation


def resolve_configured_model(
    config: Dict[str, Any],
    config_dir: Path,
    section_name: str,
    args: argparse.Namespace,
) -> Path:
    section = require_mapping(config[section_name], section_name)
    width, height = get_input_size(section, section_name)
    return resolve_model_for_npu(
        config_path_value(config, section_name, config_dir),
        width,
        height,
        str(section.get("input_name", "images")),
        args.soc_version,
        args.atc_bin,
        args.precision_mode,
        args.force_convert,
        args.om_cache_dir,
    )


def resolve_all_models(
    config: Dict[str, Any],
    config_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Path]:
    return {
        "scene": resolve_configured_model(config, config_dir, "scene_router", args),
        "easy": resolve_configured_model(config, config_dir, "easy_branch", args),
        "hard": resolve_configured_model(config, config_dir, "hard_branch", args),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-model routed NPU inference for Ascend 310B.")
    parser.add_argument("--input", type=Path, help="Image file or directory. Not required with --convert-only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ascend310bplus"))
    parser.add_argument("--summary", type=Path, help="Summary JSON path. Default: OUTPUT_DIR/summary.json.")
    parser.add_argument("--jsonl", type=Path, help="Prediction JSONL path. Default: OUTPUT_DIR/predictions.jsonl.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--soc-version", help="ATC soc_version, for example Ascend310B4. Defaults to SOC_VERSION.")
    parser.add_argument("--atc-bin", default="atc")
    parser.add_argument("--precision-mode", default=None)
    parser.add_argument("--om-cache-dir", type=Path)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--convert-only", action="store_true", help="Only convert ONNX models to OM and exit.")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--route-confidence", type=float)
    parser.add_argument("--scene-model", type=Path)
    parser.add_argument("--easy-model", type=Path)
    parser.add_argument("--hard-model", type=Path)
    parser.add_argument("--det-conf", type=float, help="Set both easy and hard detector confidence.")
    parser.add_argument("--easy-conf", type=float)
    parser.add_argument("--hard-conf", type=float)
    parser.add_argument("--hard-iou", type=float)
    parser.add_argument("--scene-score-mode", choices=["auto", "raw", "softmax"])
    parser.add_argument("--hard-score-activation", choices=["auto", "raw", "sigmoid"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_json(config_path)
    apply_cli_overrides(config, args)
    config_dir = config_path.parent

    if args.convert_only:
        model_paths = resolve_all_models(config, config_dir, args)
        model_path_text = {key: str(value) for key, value in model_paths.items()}
        print(json.dumps({"models": model_path_text}, ensure_ascii=False, indent=2))
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --convert-only is used.")

    scene_model_path = resolve_configured_model(config, config_dir, "scene_router", args)
    model_path_text: Dict[str, str] = {"scene": str(scene_model_path)}
    image_paths = iter_images(args.input)
    scene_section = require_mapping(config["scene_router"], "scene_router")
    easy_section = require_mapping(config["easy_branch"], "easy_branch")
    hard_section = require_mapping(config["hard_branch"], "hard_branch")
    global_class_names = [str(item) for item in config.get("global_classes", [])]
    if not global_class_names:
        global_class_names = get_classes(easy_section, "easy_branch")

    print("[INFO] Images: %d" % len(image_paths), flush=True)
    print("[INFO] Scene model: %s" % scene_model_path, flush=True)

    scene_results: Dict[Path, Dict[str, Any]] = {}
    easy_images: List[Path] = []
    hard_images: List[Path] = []
    rows: Dict[Path, Dict[str, Any]] = {}

    # Keep the scene model loaded while detectors are used. This keeps one ACL
    # runtime lifetime for the whole process and avoids repeated init/finalize.
    with AscendOmModel(scene_model_path, args.device_id) as scene_model:
        for index, image_path in enumerate(image_paths, start=1):
            result = run_scene_router(scene_model, image_path, scene_section)
            route, reason = choose_route(result, config)
            result["route"] = route
            result["route_reason"] = reason
            scene_results[image_path] = result
            if route == "easy":
                easy_images.append(image_path)
            elif route == "hard":
                hard_images.append(image_path)
            else:
                raise ValueError("Unsupported route: %s" % route)
            if index % 50 == 0 or index == len(image_paths):
                print("[INFO] Routed %d/%d images" % (index, len(image_paths)), flush=True)

        if easy_images:
            print("[INFO] Easy images: %d" % len(easy_images), flush=True)
            easy_model_path = resolve_configured_model(config, config_dir, "easy_branch", args)
            model_path_text["easy"] = str(easy_model_path)
            print("[INFO] Easy model : %s" % easy_model_path, flush=True)
            with AscendOmModel(easy_model_path, args.device_id) as easy_model:
                for image_path in easy_images:
                    detections, detector_info = run_easy_detector(
                        easy_model,
                        image_path,
                        easy_section,
                        args.min_box_size,
                    )
                    rows[image_path] = {
                        "detections": detections,
                        "detector": detector_info,
                    }

        if hard_images:
            print("[INFO] Hard images: %d" % len(hard_images), flush=True)
            hard_model_path = resolve_configured_model(config, config_dir, "hard_branch", args)
            model_path_text["hard"] = str(hard_model_path)
            print("[INFO] Hard model : %s" % hard_model_path, flush=True)
            with AscendOmModel(hard_model_path, args.device_id) as hard_model:
                for image_path in hard_images:
                    detections, detector_info = run_hard_detector(
                        hard_model,
                        image_path,
                        hard_section,
                        global_class_names,
                        args.min_box_size,
                    )
                    rows[image_path] = {
                        "detections": detections,
                        "detector": detector_info,
                    }

    output_rows: List[Dict[str, Any]] = []
    image_output_dir = args.output_dir / "images"
    for image_path in image_paths:
        scene = scene_results[image_path]
        detector_payload = rows[image_path]["detector"]
        detections = rows[image_path]["detections"]
        with Image.open(image_path) as image:
            image_size = [image.size[0], image.size[1]]
        row = {
            "image": str(image_path),
            "image_size": image_size,
            "scene": {
                "id": int(scene["scene_id"]),
                "name": str(scene["scene_name"]),
                "confidence": round(float(scene["confidence"]), 6),
                "scores": [round(float(item), 6) for item in scene["scores"]],
                "elapsed_ms": round(float(scene["elapsed_ms"]), 3),
            },
            "detector": {
                "route": str(scene["route"]),
                "route_reason": str(scene["route_reason"]),
                "elapsed_ms": round(float(detector_payload["elapsed_ms"]), 3),
                "input_size": detector_payload["input_size"],
                "output_count": detector_payload["output_count"],
            },
            "detections": [item.to_dict() for item in detections],
        }
        output_rows.append(row)
        if not args.no_save_images:
            save_annotated_image(
                image_path,
                relative_output_path(args.input, image_path, image_output_dir),
                detections,
                str(scene["scene_name"]),
                str(scene["route"]),
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.output_dir / "summary.json")
    jsonl_path = args.jsonl or (args.output_dir / "predictions.jsonl")
    summary = summarize_rows(output_rows, model_path_text, config)
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
    except Exception as exc:
        print("routed_infer_npu.py: %s" % exc, file=sys.stderr)
        raise
