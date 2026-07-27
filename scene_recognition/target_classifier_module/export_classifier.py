from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from scene_recognition.target_classifier_module.training import build_resnet18


def export_classifier(checkpoint_path: Path, output_dir: Path, opset: int = 12) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint.get("image_size", 224))
    model = build_resnet18(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = (output_dir / "resnet18_target_bs1.onnx").resolve()
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    task_type = checkpoint.get(
        "task_type",
        "whole_image_multilabel" if "thresholds" in checkpoint else "target_crop_single_label",
    )
    if task_type == "whole_image_multilabel":
        postprocessing = {
            "activation": "sigmoid_per_class",
            "thresholds": dict(
                zip(class_names, checkpoint.get("thresholds", [0.5] * len(class_names)))
            ),
            "note": "每类独立与阈值比较，可同时输出多个目标类别。",
        }
    else:
        postprocessing = {
            "activation": "softmax",
            "decision": "argmax",
            "note": "仅用于单个真实框裁剪目标的四选一分类。",
        }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "onnx_model": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_opset": opset,
        "input_name": "images",
        "input_shape": [1, 3, image_size, image_size],
        "output_name": "logits",
        "class_names": class_names,
        "task_type": task_type,
        "preprocessing": {
            "square_pad_fill": 0,
            "resize": [image_size, image_size],
            "scale_pixels_to": [0.0, 1.0],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "note": "ONNX只包含神经网络；部署程序必须在送入images前复现这些预处理。",
        },
        "postprocessing": postprocessing,
        "onnx_checker": "passed",
        "atc_command_template": (
            f'atc --model="{onnx_path.as_posix()}" --framework=5 '
            f'--output="resnet18_target_bs1" --input_format=NCHW '
            f'--input_shape="images:1,3,{image_size},{image_size}" '
            "--soc_version=<通过npu-smi info确认，例如Ascend310B4>"
        ),
        "deployment_status": "仅完成ONNX导出；OM转换、板卡精度和FPS仍需实机验证。",
    }
    (output_dir / "onnx_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="导出ResNet18目标分类ONNX和310B ATC模板")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    print(
        json.dumps(
            export_classifier(args.checkpoint, args.output, args.opset),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
