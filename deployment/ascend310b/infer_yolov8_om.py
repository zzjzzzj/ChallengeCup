#!/usr/bin/env python3
"""Run a YOLOv8 OM model on Ascend 310B with the CANN ACL Python API.

This runtime script is intentionally independent from the training project:
it does not import torch, torchvision, ultralytics, or the local package.
It is compatible with Python 3.9 and expects a static batch=1 YOLOv8 export.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


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
    image_size: int,
    soc_version: str,
    om_cache_dir: Optional[Path],
) -> Path:
    output_dir = om_cache_dir if om_cache_dir is not None else onnx_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / ("%s_%dx%d_%s.om" % (onnx_path.stem, image_size, image_size, soc_version))


def convert_onnx_to_om(
    onnx_path: Path,
    om_path: Path,
    image_size: int,
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
        "--input_shape=%s:1,3,%d,%d" % (input_name, image_size, image_size),
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


def resolve_model_for_npu(args: argparse.Namespace) -> Path:
    suffix = args.model.suffix.lower()
    if suffix == ".om":
        return args.model
    if suffix != ".onnx":
        raise SystemExit("Model must be .om or .onnx: %s" % args.model)
    if not args.model.is_file():
        raise FileNotFoundError("ONNX model not found: %s" % args.model)
    soc_version = resolve_soc_version(args.soc_version)
    om_path = om_path_for_onnx(args.model, args.image_size, soc_version, args.om_cache_dir)
    return convert_onnx_to_om(
        onnx_path=args.model,
        om_path=om_path,
        image_size=args.image_size,
        input_name=args.input_name,
        soc_version=soc_version,
        atc_bin=args.atc_bin,
        precision_mode=args.precision_mode,
        force=args.force_convert,
    )


@dataclass
class LetterboxMeta:
    original_width: int
    original_height: int
    scale: float
    pad_x: int
    pad_y: int


@dataclass
class Detection:
    class_id: int
    name: str
    confidence: float
    box_xyxy: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, object]:
        x1, y1, x2, y2 = self.box_xyxy
        return {
            "class_id": self.class_id,
            "name": self.name,
            "confidence": round(self.confidence, 6),
            "box_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "box_xywh": [
                round(x1, 3),
                round(y1, 3),
                round(x2 - x1, 3),
                round(y2 - y1, 3),
            ],
        }


class AclError(RuntimeError):
    pass


class AscendOmModel:
    """Small ACL wrapper for one-input, one-output-or-more static OM models."""

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
        self.context = self._value(
            self.acl.rt.create_context(device_id), "acl.rt.create_context"
        )
        self.model_id = self._value(
            self.acl.mdl.load_from_file(str(model_path)), "acl.mdl.load_from_file"
        )
        self.model_desc = self.acl.mdl.create_desc()
        self._check(
            self.acl.mdl.get_desc(self.model_desc, self.model_id), "acl.mdl.get_desc"
        )

        input_count = int(self._value(self.acl.mdl.get_num_inputs(self.model_desc), "acl.mdl.get_num_inputs"))
        if input_count != 1:
            raise RuntimeError("This runtime expects exactly one model input.")
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
            self.acl.mdl.add_dataset_buffer(
                self.input_dataset, self.input_data_buffer
            ),
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
                    "ACL runtime is already initialized on device %s, cannot use device %s"
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
                "Input tensor byte size mismatch: got %d, model expects %d"
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

        outputs = []
        for output_ptr, output_size, _data_buffer in self.output_buffers:
            host_output = self._value(
                self.acl.rt.malloc_host(output_size), "acl.rt.malloc_host(output)"
            )
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


def load_metadata(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_names(metadata: Dict[str, object], classes_path: Optional[Path]) -> List[str]:
    names = metadata.get("class_names")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    if classes_path is not None:
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError("Class names are missing. Pass --metadata or --classes.")


def iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        images = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            return images
    raise FileNotFoundError("No image file found at %s" % path)


def letterbox(image: Image.Image, image_size: int) -> Tuple[np.ndarray, LetterboxMeta]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(float(image_size) / float(width), float(image_size) / float(height))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = rgb.resize((resized_width, resized_height), Image.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(array), LetterboxMeta(width, height, scale, pad_x, pad_y)


def reshape_yolov8_output(
    flat_output: np.ndarray,
    metadata: Dict[str, object],
    class_count: int,
    layout: str,
) -> np.ndarray:
    shape = metadata.get("output_shape")
    if isinstance(shape, list) and shape:
        expected = int(np.prod([int(value) for value in shape]))
        if expected == flat_output.size:
            output = flat_output.reshape([int(value) for value in shape])
        else:
            output = flat_output
    else:
        output = flat_output

    channel_count = 4 + class_count
    if output.ndim == 1:
        if output.size % channel_count != 0:
            raise ValueError(
                "Cannot reshape model output of size %d for %d classes"
                % (output.size, class_count)
            )
        if layout == "channels-last":
            return output.reshape(-1, channel_count)
        return output.reshape(channel_count, -1).T

    output = np.squeeze(output)
    if output.ndim == 2:
        if output.shape[0] == channel_count and layout != "channels-last":
            return output.T
        if output.shape[1] == channel_count:
            return output
    raise ValueError("Unsupported YOLO output shape after squeeze: %s" % (output.shape,))


def clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def box_iou(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    kept: List[Detection] = []
    by_class: Dict[int, List[Detection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)

    for class_detections in by_class.values():
        pending = sorted(class_detections, key=lambda item: item.confidence, reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [
                candidate
                for candidate in pending
                if box_iou(best.box_xyxy, candidate.box_xyxy) <= iou_threshold
            ]
    return sorted(kept, key=lambda item: item.confidence, reverse=True)


def decode_yolov8(
    output: np.ndarray,
    metadata: Dict[str, object],
    class_names: Sequence[str],
    letterbox_meta: LetterboxMeta,
    confidence_threshold: float,
    iou_threshold: float,
    layout: str,
) -> List[Detection]:
    rows = reshape_yolov8_output(output, metadata, len(class_names), layout)
    candidates: List[Detection] = []
    for row in rows:
        class_scores = row[4 : 4 + len(class_names)]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue

        x_center, y_center, width, height = [float(value) for value in row[:4]]
        x1 = (x_center - width / 2.0 - letterbox_meta.pad_x) / letterbox_meta.scale
        y1 = (y_center - height / 2.0 - letterbox_meta.pad_y) / letterbox_meta.scale
        x2 = (x_center + width / 2.0 - letterbox_meta.pad_x) / letterbox_meta.scale
        y2 = (y_center + height / 2.0 - letterbox_meta.pad_y) / letterbox_meta.scale
        x1 = clip(x1, 0.0, float(letterbox_meta.original_width))
        y1 = clip(y1, 0.0, float(letterbox_meta.original_height))
        x2 = clip(x2, 0.0, float(letterbox_meta.original_width))
        y2 = clip(y2, 0.0, float(letterbox_meta.original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            Detection(
                class_id=class_id,
                name=str(class_names[class_id]),
                confidence=confidence,
                box_xyxy=(x1, y1, x2, y2),
            )
        )
    return nms(candidates, iou_threshold)


def save_annotated_image(image_path: Path, detections: Sequence[Detection], output_path: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        label = "%s %.2f" % (detection.name, detection.confidence)
        draw.text((x1 + 2, max(0.0, y1 - 12)), label, fill=(255, 64, 64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output_path)


def resolve_annotation_path(save_image: Path, image_path: Path, multi_image: bool) -> Path:
    if multi_image or save_image.suffix == "" or save_image.is_dir():
        save_image.mkdir(parents=True, exist_ok=True)
        return save_image / ("%s_detected.png" % image_path.stem)
    return save_image


def run_image(
    model: AscendOmModel,
    image_path: Path,
    metadata: Dict[str, object],
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    with Image.open(image_path) as image:
        tensor, transform = letterbox(image, args.image_size)
    output_dtype = np.dtype(args.output_dtype)
    outputs = model.infer(tensor.astype(np.float32), output_dtype)
    if not outputs:
        raise RuntimeError("Model produced no output buffers.")
    detections = decode_yolov8(
        outputs[0],
        metadata,
        class_names,
        transform,
        args.confidence,
        args.iou,
        args.output_layout,
    )
    if args.save_image is not None:
        save_annotated_image(
            image_path,
            detections,
            resolve_annotation_path(args.save_image, image_path, args.multi_image),
        )
    return {
        "image": str(image_path),
        "image_size": [transform.original_width, transform.original_height],
        "detections": [detection.to_dict() for detection in detections],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 OM inference on Ascend 310B.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .om model or a .onnx model to auto-convert.")
    parser.add_argument("--image", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--metadata", type=Path, default=Path("package_metadata.json"))
    parser.add_argument("--classes", type=Path, help="Fallback classes.txt when metadata is unavailable.")
    parser.add_argument("--output", type=Path, help="JSON result path. Defaults to stdout.")
    parser.add_argument("--save-image", type=Path, help="Optional annotated image file or directory.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--soc-version", help="ATC soc_version, for example Ascend310B4. Defaults to SOC_VERSION env.")
    parser.add_argument("--atc-bin", default="atc")
    parser.add_argument("--precision-mode", default=None, help="Optional ATC precision_mode, for example allow_fp32_to_fp16.")
    parser.add_argument("--om-cache-dir", type=Path, help="Directory for auto-converted OM files. Defaults to ONNX directory.")
    parser.add_argument("--force-convert", action="store_true", help="Re-run ATC even when the cached OM already exists.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--output-dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--output-layout", default="channels-first", choices=["channels-first", "channels-last"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metadata = load_metadata(args.metadata if args.metadata.is_file() else None)
    class_names = load_class_names(metadata, args.classes)
    if "image_size" in metadata:
        args.image_size = int(metadata["image_size"])
    if "output_layout" in metadata:
        args.output_layout = str(metadata["output_layout"])
    model_path = resolve_model_for_npu(args)

    image_paths = iter_images(args.image)
    args.multi_image = len(image_paths) > 1
    with AscendOmModel(model_path, args.device_id) as model:
        results = [
            run_image(model, image_path, metadata, class_names, args)
            for image_path in image_paths
        ]

    payload: object = results if len(results) > 1 else results[0]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("infer_yolov8_om.py: %s" % exc, file=sys.stderr)
        raise
