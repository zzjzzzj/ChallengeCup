#!/usr/bin/env python3
"""Probe board-side training dependencies and device visibility."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from typing import Any, Dict


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def probe() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "modules": {
            "torch": module_available("torch"),
            "torchvision": module_available("torchvision"),
            "ultralytics": module_available("ultralytics"),
            "torch_npu": module_available("torch_npu"),
        },
        "torch": None,
        "npu": None,
        "recommendation": [],
    }
    if not result["modules"]["torch"]:
        result["recommendation"].append("Install CPU PyTorch for board-side training, or use --augment-only.")
        return result

    import torch  # type: ignore

    result["torch"] = {
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0,
        "thread_count": int(torch.get_num_threads()),
    }
    if result["modules"]["torch_npu"]:
        try:
            import torch_npu  # type: ignore  # noqa: F401

            npu = getattr(torch, "npu", None)
            result["npu"] = {
                "torch_npu_imported": True,
                "torch_has_npu": npu is not None,
                "is_available": bool(npu.is_available()) if npu is not None and hasattr(npu, "is_available") else False,
                "device_count": int(npu.device_count()) if npu is not None and hasattr(npu, "device_count") else 0,
            }
        except Exception as error:  # pragma: no cover - board-only diagnostic
            result["npu"] = {"torch_npu_imported": False, "error": repr(error)}
    else:
        result["npu"] = {"torch_npu_imported": False}

    if not result["modules"]["ultralytics"]:
        result["recommendation"].append("Install ultralytics to run train.py yolo/continual-yolo/class-il-yolo.")
    if result["npu"] and not result["npu"].get("is_available"):
        result["recommendation"].append(
            "Use --device cpu for training on Ascend 310B boards; keep NPU for OM/ACL inference."
        )
    return result


def main() -> int:
    print(json.dumps(probe(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
