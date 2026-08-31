#!/usr/bin/env python3
"""串联离线增广、已训练模型推理和结果归档的 310B 部署流程。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ascend310b_end_to_end"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from result_formatter import summarize_prediction_payload, summary_csv_row, write_summary_csv


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行 310B 端到端流程：数据增广 -> 已训练模型推理 -> 结果输出。"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path, help="YOLO 数据集根目录。")
    source.add_argument("--images", type=Path, help="源训练图像目录；需同时传入 --labels。")
    parser.add_argument("--labels", type=Path, help="与 --images 配对的 YOLO 标签目录。")
    parser.add_argument("--split", default="train", help="--dataset-root 中的训练划分名，默认 train。")
    parser.add_argument("--classes", type=Path, help="类别名称文件；默认读取 <dataset-root>/classes.txt。")
    parser.add_argument("--val-images", type=Path, help="独立验证图像目录或图像文件。")
    parser.add_argument("--val-root", type=Path, help="独立验证 YOLO 数据集根目录。")
    parser.add_argument("--val-split", default="val", help="--val-root 中的验证划分名，默认 val。")
    parser.add_argument(
        "--default-modality",
        choices=["ir", "sar"],
        help="当图片名不以 ir_ 或 sar_ 开头时使用的默认模态。",
    )
    parser.add_argument(
        "--no-include-original",
        dest="include_original",
        action="store_false",
        default=True,
        help="增广集中不保留原始图片，仅保留增广版本。",
    )

    parser.add_argument("--model", type=Path, required=True, help="已训练并导出的 .onnx 或 .om 模型。")
    parser.add_argument(
        "--backend",
        choices=["auto", "onnx", "om"],
        default="auto",
        help="推理后端；auto 根据模型扩展名选择。",
    )
    parser.add_argument(
        "--infer-input",
        type=Path,
        help="待推理图片或目录；未传入时对本次生成的增广图片推理。",
    )
    parser.add_argument("--metadata", type=Path, help="模型 package_metadata.json，可选。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="本次流程的全新输出目录。")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--output-layout", choices=["channels-first", "channels-last"], default="channels-first")

    parser.add_argument("--device-id", type=int, default=0, help="OM 后端使用的 NPU 设备编号。")
    parser.add_argument("--input-name", default="images", help="OM 模型输入节点名称。")
    parser.add_argument("--soc-version", help="OM 自动转换时的 ATC soc_version。")
    parser.add_argument("--atc-bin", default="atc", help="ATC 可执行程序。")
    parser.add_argument("--precision-mode", help="可选的 ATC precision_mode。")
    parser.add_argument("--om-cache-dir", type=Path, help="ONNX 自动转换 OM 时的缓存目录。")
    parser.add_argument("--force-convert", action="store_true", help="强制重新把 ONNX 转换为 OM。")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float32")
    return parser.parse_args(argv)


def resolve_backend(model_path: Path, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".om":
        return "om"
    raise ValueError(
        "无法根据模型扩展名选择推理后端：%s。请传入 .onnx/.om 模型，或显式指定 --backend。"
        % model_path
    )


def ensure_new_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("--output-dir 不是目录：%s" % output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "输出目录已存在且非空：%s。为避免覆盖已有结果，请指定新的 --output-dir。" % output_dir
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_existing_file(path: Path, option_name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError("%s 不存在或不是文件：%s" % (option_name, resolved))
    return resolved


def validate_args(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    if (args.images is None) != (args.labels is None):
        raise ValueError("--images 和 --labels 必须同时传入。")
    if args.val_images is not None and args.val_root is not None:
        raise ValueError("--val-images 和 --val-root 只能传入其中一个。")
    if args.image_size <= 0:
        raise ValueError("--image-size 必须大于 0。")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence 必须在 [0, 1] 范围内。")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou 必须在 (0, 1] 范围内。")
    model_path = resolve_existing_file(args.model, "--model")
    metadata_path = None
    if args.metadata is not None:
        metadata_path = resolve_existing_file(args.metadata, "--metadata")
    if args.infer_input is not None and not args.infer_input.resolve().exists():
        raise FileNotFoundError("--infer-input 不存在：%s" % args.infer_input.resolve())
    return model_path, metadata_path


def build_augmentation_command(args: argparse.Namespace, augmented_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(DEPLOYMENT_DIR / "augment_selected_yolo.py"),
        "--output",
        str(augmented_dir),
        "--include-original",
    ]
    if args.images is not None:
        command.extend(["--images", str(args.images.resolve()), "--labels", str(args.labels.resolve())])
    else:
        command.extend(["--dataset-root", str(args.dataset_root.resolve()), "--split", args.split])
    if args.classes is not None:
        command.extend(["--classes", str(args.classes.resolve())])
    if args.val_images is not None:
        command.extend(["--val-images", str(args.val_images.resolve())])
    if args.val_root is not None:
        command.extend(["--val-root", str(args.val_root.resolve()), "--val-split", args.val_split])
    if args.default_modality is not None:
        command.extend(["--default-modality", args.default_modality])
    if not args.include_original:
        command.remove("--include-original")
    return command


def build_inference_command(
    args: argparse.Namespace,
    backend: str,
    model_path: Path,
    metadata_path: Optional[Path],
    classes_path: Path,
    infer_input: Path,
    prediction_path: Path,
    annotated_dir: Path,
) -> List[str]:
    script_name = "infer_yolov8_onnx.py" if backend == "onnx" else "infer_yolov8_om.py"
    command = [
        sys.executable,
        str(DEPLOYMENT_DIR / script_name),
        "--model",
        str(model_path),
        "--image",
        str(infer_input),
        "--classes",
        str(classes_path),
        "--output",
        str(prediction_path),
        "--save-image",
        str(annotated_dir),
        "--image-size",
        str(args.image_size),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
        "--output-layout",
        args.output_layout,
    ]
    if metadata_path is not None:
        command.extend(["--metadata", str(metadata_path)])
    if backend == "om":
        command.extend(
            [
                "--device-id",
                str(args.device_id),
                "--input-name",
                args.input_name,
                "--atc-bin",
                args.atc_bin,
                "--output-dtype",
                args.output_dtype,
            ]
        )
        if args.soc_version is not None:
            command.extend(["--soc-version", args.soc_version])
        if args.precision_mode is not None:
            command.extend(["--precision-mode", args.precision_mode])
        if args.om_cache_dir is not None:
            command.extend(["--om-cache-dir", str(args.om_cache_dir.resolve())])
        if args.force_convert:
            command.append("--force-convert")
    return command


def run_command(command: Sequence[str]) -> None:
    print("[INFO] " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def read_json_object(path: Path, description: str) -> object:
    if not path.is_file():
        raise FileNotFoundError("%s 未生成：%s" % (description, path))
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metadata_classes(metadata_path: Optional[Path], classes_path: Path) -> None:
    if metadata_path is None:
        return
    metadata = read_json_object(metadata_path, "模型元数据")
    if not isinstance(metadata, dict):
        raise ValueError("模型元数据必须是 JSON 对象：%s" % metadata_path)
    metadata_names = metadata.get("class_names")
    if metadata_names is None:
        return
    if not isinstance(metadata_names, list) or not metadata_names:
        raise ValueError("模型元数据中的 class_names 必须是非空数组：%s" % metadata_path)
    dataset_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [str(item) for item in metadata_names] != dataset_names:
        raise ValueError(
            "模型元数据类别与本次增广数据集类别不一致。\n"
            "模型：%s\n数据：%s" % (metadata_names, dataset_names)
        )


def build_default_metadata(classes_path: Path, output_path: Path, args: argparse.Namespace) -> Path:
    class_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = {
        "schema_version": "ascend310b-pipeline-metadata-v1",
        "class_names": class_names,
        "image_size": args.image_size,
        "output_layout": args.output_layout,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def prediction_stats(payload: object) -> Dict[str, int]:
    rows = payload if isinstance(payload, list) else [payload]
    valid_rows = [row for row in rows if isinstance(row, dict)]
    detection_count = sum(
        len(row.get("detections", []))
        for row in valid_rows
        if isinstance(row.get("detections", []), list)
    )
    return {"images": len(valid_rows), "detections": detection_count}


def make_summary(
    started_at: str,
    elapsed_seconds: float,
    output_dir: Path,
    backend: str,
    model_path: Path,
    metadata_path: Optional[Path],
    augmented_dir: Path,
    augmentation_summary: object,
    infer_input: Path,
    prediction_path: Path,
    annotated_dir: Path,
    predictions: object,
    result_summary_path: Path,
    result_summaries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": "ascend310b-end-to-end-v1",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "training": {
            "performed": False,
            "reason": "本流程复用已训练模型，仅执行数据增广和推理。",
        },
        "model": {
            "path": str(model_path),
            "backend": backend,
            "metadata": str(metadata_path) if metadata_path is not None else None,
        },
        "augmentation": {
            "output_dir": str(augmented_dir),
            "summary": augmentation_summary,
        },
        "inference": {
            "input": str(infer_input),
            "predictions": str(prediction_path),
            "annotated_images": str(annotated_dir),
            "statistics": prediction_stats(predictions),
            "result_summary_csv": str(result_summary_path),
            "result_summaries": list(result_summaries),
        },
        "output_dir": str(output_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_path, metadata_path = validate_args(args)
    backend = resolve_backend(model_path, args.backend)
    output_dir = ensure_new_output_dir(args.output_dir)
    augmented_dir = output_dir / "augmented_dataset"
    inference_dir = output_dir / "inference"
    prediction_path = inference_dir / "predictions.json"
    annotated_dir = inference_dir / "annotated_images"
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_clock = monotonic()

    run_command(build_augmentation_command(args, augmented_dir))
    classes_path = augmented_dir / "classes.txt"
    augmentation_summary_path = augmented_dir / "augmentation_summary.json"
    augmentation_summary = read_json_object(augmentation_summary_path, "增广摘要")
    if not classes_path.is_file():
        raise FileNotFoundError("增广数据集未生成类别文件：%s" % classes_path)
    validate_metadata_classes(metadata_path, classes_path)
    effective_metadata_path = metadata_path or build_default_metadata(
        classes_path,
        inference_dir / "runtime_metadata.json",
        args,
    )

    infer_input = args.infer_input.resolve() if args.infer_input is not None else augmented_dir / "images"
    if not infer_input.exists():
        raise FileNotFoundError("待推理输入不存在：%s" % infer_input)
    run_command(
        build_inference_command(
            args,
            backend,
            model_path,
            effective_metadata_path,
            classes_path,
            infer_input,
            prediction_path,
            annotated_dir,
        )
    )
    predictions = read_json_object(prediction_path, "推理结果")
    result_summaries = summarize_prediction_payload(predictions)
    result_summary_path = inference_dir / "result_summary.csv"
    write_summary_csv(
        result_summary_path,
        [summary_csv_row(item["image"], item) for item in result_summaries],
    )
    summary = make_summary(
        started_at,
        monotonic() - started_clock,
        output_dir,
        backend,
        model_path,
        effective_metadata_path,
        augmented_dir,
        augmentation_summary,
        infer_input,
        prediction_path,
        annotated_dir,
        predictions,
        result_summary_path,
        result_summaries,
    )
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[INFO] 增广数据集：%s" % augmented_dir, flush=True)
    print("[INFO] 推理结果：%s" % prediction_path, flush=True)
    print("[INFO] 结果摘要：%s" % result_summary_path, flush=True)
    print("[INFO] 标注图片：%s" % annotated_dir, flush=True)
    print("[INFO] 流程汇总：%s" % summary_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("run_end_to_end.py: %s" % exc, file=sys.stderr)
        raise
