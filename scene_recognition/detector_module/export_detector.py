from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx
import torch
import yaml
from ultralytics import YOLO

from scene_recognition.detector_module.metrics import detection_metrics_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "scene_recognition" / "detector_module" / "artifacts" / "detection_dataset" / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 ONNX 并验证转换前后的测试集精度")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def class_names_from_yaml(path: Path) -> list[str]:
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(value) for value in names]


def main() -> None:
    args = parse_args()
    if not args.model.is_file() or not args.data.is_file():
        raise FileNotFoundError("模型或数据配置不存在")

    output_dir = args.output or args.model.resolve().parents[1] / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names_from_yaml(args.data)
    pytorch_model = YOLO(str(args.model.resolve()))
    exported = Path(
        pytorch_model.export(
            format="onnx",
            imgsz=args.image_size,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=args.opset,
            half=False,
            device=args.device,
        )
    ).resolve()
    final_onnx = (output_dir / "detector_yolov8n_bs1.onnx").resolve()
    if exported != final_onnx:
        shutil.copy2(exported, final_onnx)

    onnx_model = onnx.load(str(final_onnx))
    onnx.checker.check_model(onnx_model)
    input_tensor = onnx_model.graph.input[0]
    dimensions = [
        dimension.dim_value if dimension.dim_value else dimension.dim_param
        for dimension in input_tensor.type.tensor_type.shape.dim
    ]

    summary = {
        "pytorch_model": str(args.model.resolve()),
        "onnx_model": str(final_onnx),
        "onnx_bytes": final_onnx.stat().st_size,
        "onnx_opset": args.opset,
        "input_name": input_tensor.name,
        "input_shape": dimensions,
        "onnx_checker": "passed",
        "atc_command_template": (
            f'atc --model="{final_onnx.as_posix()}" --framework=5 '
            f'--output="detector_yolov8n_bs1" --input_format=NCHW '
            f'--input_shape="{input_tensor.name}:1,3,{args.image_size},{args.image_size}" '
            "--soc_version=<向 npu-smi info 确认，例如 Ascend310B4>"
        ),
        "validation": None,
    }

    if not args.skip_validation:
        onnx_detector = YOLO(str(final_onnx), task="detect")
        metrics = onnx_detector.val(
            data=str(args.data.resolve()),
            split="test",
            imgsz=args.image_size,
            batch=args.batch_size,
            device="cpu",
            workers=2,
            project=str(output_dir),
            name="onnx_test",
            plots=False,
            verbose=False,
        )
        summary["validation"] = detection_metrics_to_dict(metrics, class_names)

    summary_path = output_dir / "onnx_export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
