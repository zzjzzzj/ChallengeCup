#!/usr/bin/env python3
"""Print a small Ascend 310B runtime environment report."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any, Dict, List


def run_command(command: List[str]) -> Dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    completed = subprocess.run(
        [executable, *command[1:]],
        text=True,
        capture_output=True,
    )
    return {
        "available": True,
        "command": [executable, *command[1:]],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip().splitlines()[:20],
        "stderr": completed.stderr.strip().splitlines()[:20],
    }


def package_version(package_name: str) -> object:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def import_status(module_name: str, package_name: str = "") -> Dict[str, Any]:
    try:
        module = __import__(module_name)
        return {
            "ok": True,
            "module": str(module),
            "version": package_version(package_name or module_name),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    report: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "atc": run_command(["atc", "--version"]),
        "npu_smi": run_command(["npu-smi", "info"]),
    }
    try:
        import acl  # type: ignore

        report["acl_import"] = {"ok": True, "module": str(acl)}
    except Exception as exc:
        report["acl_import"] = {"ok": False, "error": str(exc)}
    try:
        import numpy as np

        report["numpy"] = np.__version__
    except Exception as exc:
        report["numpy"] = {"error": str(exc)}
    try:
        import PIL

        report["pillow"] = PIL.__version__
    except Exception as exc:
        report["pillow"] = {"error": str(exc)}
    report["attrs_attr_import"] = import_status("attr", "attrs")
    report["tornado_import"] = import_status("tornado")
    report["absl_import"] = import_status("absl", "absl-py")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
