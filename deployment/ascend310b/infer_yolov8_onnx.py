#!/usr/bin/env python3
"""Run a YOLOv8 ONNX model with ONNX Runtime CPU.

This script shares the same preprocessing, postprocessing, JSON output, and
annotated-image format as infer_yolov8_om.py. It is useful as the first
end-to-end runtime on an aarch64 board before switching the backend to OM.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image

from infer_yolov8_om import (
    decode_yolov8,
    iter_images,
    letterbox,
    load_class_names,
    load_metadata,
    resolve_annotation_path,
    save_annotated_image,
)


class OnnxCpuModel:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import onnxruntime. Install it with: "
                "python3 -m pip install onnxruntime"
            ) from exc
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError("This runtime expects exactly one ONNX input.")
        self.input_name = inputs[0].name

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: np.ascontiguousarray(tensor)})
        if not outputs:
            raise RuntimeError("ONNX model produced no outputs.")
        return np.asarray(outputs[0])


def run_image(
    model: OnnxCpuModel,
    image_path: Path,
    metadata: Dict[str, object],
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    with Image.open(image_path) as image:
        tensor, transform = letterbox(image, args.image_size)
    output = model.infer(tensor.astype(np.float32))
    detections = decode_yolov8(
        output,
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
    parser = argparse.ArgumentParser(description="Run YOLOv8 ONNX inference on CPU.")
    parser.add_argument("--model", type=Path, required=True, help="Path to detector_yolov8n_bs1.onnx.")
    parser.add_argument("--image", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--metadata", type=Path, default=Path("package_metadata.json"))
    parser.add_argument("--classes", type=Path, help="Fallback classes.txt when metadata is unavailable.")
    parser.add_argument("--output", type=Path, help="JSON result path. Defaults to stdout.")
    parser.add_argument("--save-image", type=Path, help="Optional annotated image file or directory.")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
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

    image_paths = iter_images(args.image)
    args.multi_image = len(image_paths) > 1
    model = OnnxCpuModel(args.model)
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
        print("infer_yolov8_onnx.py: %s" % exc, file=sys.stderr)
        raise
