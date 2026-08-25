#!/usr/bin/env python3
"""Build a portable Ascend 310B inference package for this project."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = Path(__file__).resolve().parent


def read_class_names(data_yaml: Optional[Path], classes_path: Optional[Path]) -> List[str]:
    if classes_path is not None:
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if data_yaml is None:
        raise ValueError("Pass --data or --classes so the runtime package knows class names.")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when reading class names from --data.") from exc
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def run_export(checkpoint: Path, data_yaml: Path, output_dir: Path, image_size: int, opset: int) -> Path:
    command = [
        sys.executable,
        "-m",
        "scene_recognition.detector_module.export_detector",
        "--model",
        str(checkpoint),
        "--data",
        str(data_yaml),
        "--output",
        str(output_dir),
        "--image-size",
        str(image_size),
        "--opset",
        str(opset),
        "--skip-validation",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    onnx_path = output_dir / "detector_yolov8n_bs1.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError("ONNX export did not produce %s" % onnx_path)
    return onnx_path


def copy_runtime_files(output_dir: Path) -> None:
    for name in [
        "infer_yolov8_om.py",
        "convert_onnx_to_om.sh",
        "run_infer.sh",
        "requirements-runtime.txt",
    ]:
        shutil.copy2(DEPLOYMENT_DIR / name, output_dir / name)


def make_metadata(
    args: argparse.Namespace,
    class_names: Sequence[str],
    onnx_file: str,
) -> Dict[str, object]:
    channel_count = 4 + len(class_names)
    grid_points = sum((args.image_size // stride) ** 2 for stride in (8, 16, 32))
    return {
        "package_created_at": datetime.now().isoformat(timespec="seconds"),
        "target": "Ascend 310B aarch64 CANN ACL",
        "model_family": "YOLOv8 detection",
        "source_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "onnx_file": onnx_file,
        "expected_om_file": "detector_yolov8n_%d_bs1.om" % args.image_size,
        "image_size": args.image_size,
        "input_name": args.input_name,
        "input_shape": [1, 3, args.image_size, args.image_size],
        "input_dtype": "float32",
        "input_nbytes": 1 * 3 * args.image_size * args.image_size * 4,
        "output_layout": "channels-first",
        "output_shape": [1, channel_count, grid_points],
        "output_dtype": "float32",
        "class_names": list(class_names),
        "preprocess": {
            "letterbox_fill": [114, 114, 114],
            "color": "RGB",
            "scale": "divide_by_255",
            "layout": "NCHW",
        },
        "postprocess": {
            "confidence_threshold": args.confidence,
            "nms_iou_threshold": args.iou,
            "nms": "class-aware CPU NMS in infer_yolov8_om.py",
        },
        "atc": {
            "framework": 5,
            "input_format": "NCHW",
            "precision_mode": "allow_fp32_to_fp16",
            "soc_version": "<set SOC_VERSION on the Ascend device>",
        },
    }


def write_runtime_readme(output_dir: Path, args: argparse.Namespace) -> None:
    text = """# Ascend 310B Runtime Package

This folder is intended to be copied to the aarch64 Ascend 310B device.

## On the Ascend device

```bash
cd {package_name}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r requirements-runtime.txt

npu-smi info
atc --list_soc_version
export SOC_VERSION=Ascend310B4
bash convert_onnx_to_om.sh detector_yolov8n_bs1.onnx detector_yolov8n_{image_size}_bs1

python3 infer_yolov8_om.py \\
  --model detector_yolov8n_{image_size}_bs1.om \\
  --image demo.png \\
  --metadata package_metadata.json \\
  --output result.json \\
  --save-image outputs
```

Replace `Ascend310B4` with the exact value reported by your device.
The Python runtime does not need PyTorch or Ultralytics.
""".format(
        package_name=output_dir.name,
        image_size=args.image_size,
    )
    (output_dir / "README_RUNTIME.md").write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Ascend 310B runtime package.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="YOLO .pt checkpoint to export.")
    source.add_argument("--onnx", type=Path, help="Existing static batch=1 ONNX file.")
    parser.add_argument("--data", type=Path, help="YOLO data YAML, used for class names and export.")
    parser.add_argument("--classes", type=Path, help="classes.txt fallback when using --onnx.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dist" / "ascend310b_yolov8n")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--archive", action="store_true", help="Also create a .zip next to the output folder.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output folder.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.checkpoint is not None and args.data is None:
        raise SystemExit("--data is required when exporting from --checkpoint.")

    output_dir = args.output.resolve()
    model_dir = output_dir / "model"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(
            "Output folder already exists and is not empty: %s\n"
            "Pass --force to overwrite it." % output_dir
        )
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    model_dir.mkdir(parents=True)

    class_names = read_class_names(args.data, args.classes)
    if args.checkpoint is not None:
        onnx_path = run_export(
            args.checkpoint.resolve(),
            args.data.resolve(),
            model_dir,
            args.image_size,
            args.opset,
        )
    else:
        onnx_path = model_dir / "detector_yolov8n_bs1.onnx"
        shutil.copy2(args.onnx.resolve(), onnx_path)

    copy_runtime_files(output_dir)
    shutil.copy2(onnx_path, output_dir / "detector_yolov8n_bs1.onnx")
    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    metadata = make_metadata(args, class_names, "detector_yolov8n_bs1.onnx")
    (output_dir / "package_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_runtime_readme(output_dir, args)

    if args.archive:
        archive_base = str(output_dir)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=output_dir)
        print("Created package: %s" % output_dir)
        print("Created archive: %s" % archive_path)
    else:
        print("Created package: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
