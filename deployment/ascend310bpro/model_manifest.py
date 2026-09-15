#!/usr/bin/env python3
"""Build and verify the routed deployment model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "route_config.yaml"
DEFAULT_MANIFEST = SCRIPT_DIR / "model_manifest.json"
ROLES = ("scene_router", "easy_branch", "hard_branch")


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("config must be an object: %s" % path)
    return data


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object: %s" % path)
    return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def candidate_model_paths(path_value: str, base_dir: Path) -> List[Path]:
    path = Path(path_value)
    if path.is_absolute():
        return [path.resolve()]
    candidates = [
        (base_dir / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def first_existing(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def get_section(config: Dict[str, Any], role: str) -> Dict[str, Any]:
    section = config.get(role)
    if not isinstance(section, dict):
        raise ValueError("%s must be an object in the config." % role)
    if "model" not in section:
        raise ValueError("%s.model is required." % role)
    return section


def input_size(section: Dict[str, Any], role: str) -> List[int]:
    size = section.get("input_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("%s.input_size must be [width, height]." % role)
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("%s.input_size must contain positive integers." % role)
    return [width, height]


def class_names(section: Dict[str, Any], role: str) -> List[str]:
    classes = section.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("%s.classes must be a non-empty list." % role)
    return [str(item) for item in classes]


def file_artifact(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "path": display_path(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        payload["bytes"] = int(path.stat().st_size)
        payload["sha256"] = sha256_file(path)
    return payload


def expected_om_path(model_path: Path, width: int, height: int, soc_version: str, cache_dir: Optional[Path]) -> Path:
    output_dir = cache_dir.resolve() if cache_dir is not None else model_path.parent
    return output_dir / ("%s_%dx%d_%s.om" % (model_path.stem, width, height, soc_version))


def section_manifest(
    config: Dict[str, Any],
    config_dir: Path,
    role: str,
    soc_version: Optional[str],
    om_cache_dir: Optional[Path],
) -> Dict[str, Any]:
    section = get_section(config, role)
    width, height = input_size(section, role)
    candidates = candidate_model_paths(str(section["model"]), config_dir)
    model_path = first_existing(candidates)
    payload: Dict[str, Any] = {
        "role": role,
        "configured_model": str(section["model"]),
        "searched": [display_path(path) for path in candidates],
        "model": file_artifact(model_path),
        "input_name": str(section.get("input_name", "images")),
        "input_size": [width, height],
        "classes": class_names(section, role),
        "output_dtype": str(section.get("output_dtype", "float32")),
    }

    for key in (
        "preprocess",
        "easy_scenes",
        "output_mode",
        "nms_format",
        "coords",
        "apply_nms",
        "raw_layout",
        "box_format",
        "score_activation",
        "score_mode",
        "class_id_remap",
        "confidence",
        "iou",
    ):
        if key in section:
            payload[key] = section[key]

    if soc_version and model_path.suffix.lower() == ".onnx":
        payload["cached_om"] = file_artifact(expected_om_path(model_path, width, height, soc_version, om_cache_dir))
    return payload


def build_manifest(
    config_path: Path,
    soc_version: Optional[str],
    om_cache_dir: Optional[Path],
) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    artifacts = [
        section_manifest(config, config_path.parent, role, soc_version, om_cache_dir)
        for role in ROLES
    ]
    return {
        "manifest_version": "ascend310bpro-routed-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": display_path(config_path),
        "soc_version": soc_version,
        "route_confidence": float(config.get("route_confidence", 0.60)),
        "uncertain_route": str(config.get("uncertain_route", "hard")),
        "global_classes": [str(item) for item in config.get("global_classes", [])],
        "artifacts": artifacts,
    }


def compare_equal(label: str, expected: Any, actual: Any, failures: List[str]) -> None:
    if expected != actual:
        failures.append("%s mismatch: manifest=%r current=%r" % (label, expected, actual))


def artifact_by_role(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest.artifacts must be a list.")
    result: Dict[str, Dict[str, Any]] = {}
    for item in artifacts:
        if isinstance(item, dict) and "role" in item:
            result[str(item["role"])] = item
    return result


def iter_required_file_checks(role: str, artifact: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    model = artifact.get("model")
    if isinstance(model, dict):
        yield "%s.model" % role, model
    cached_om = artifact.get("cached_om")
    if isinstance(cached_om, dict) and cached_om.get("sha256"):
        yield "%s.cached_om" % role, cached_om


def verify_manifest(config_path: Path, manifest_path: Path, om_cache_dir: Optional[Path]) -> Dict[str, Any]:
    config_path = config_path.resolve()
    manifest = read_json(manifest_path.resolve())
    current = build_manifest(config_path, str(manifest.get("soc_version") or "") or None, om_cache_dir)
    current_by_role = artifact_by_role(current)
    manifest_by_role = artifact_by_role(manifest)
    failures: List[str] = []

    compare_equal("manifest_version", manifest.get("manifest_version"), current.get("manifest_version"), failures)
    compare_equal("route_confidence", manifest.get("route_confidence"), current.get("route_confidence"), failures)
    compare_equal("uncertain_route", manifest.get("uncertain_route"), current.get("uncertain_route"), failures)
    compare_equal("global_classes", manifest.get("global_classes"), current.get("global_classes"), failures)

    for role in ROLES:
        expected = manifest_by_role.get(role)
        actual = current_by_role.get(role)
        if expected is None:
            failures.append("missing manifest artifact for %s" % role)
            continue
        if actual is None:
            failures.append("missing current artifact for %s" % role)
            continue
        for key in (
            "configured_model",
            "input_name",
            "input_size",
            "classes",
            "output_dtype",
            "preprocess",
            "easy_scenes",
            "output_mode",
            "nms_format",
            "coords",
            "apply_nms",
            "raw_layout",
            "box_format",
            "score_activation",
            "score_mode",
            "class_id_remap",
            "confidence",
            "iou",
        ):
            if key in expected or key in actual:
                compare_equal("%s.%s" % (role, key), expected.get(key), actual.get(key), failures)
        for label, expected_file in iter_required_file_checks(role, expected):
            actual_file = actual.get(label.split(".")[-1])
            if not isinstance(actual_file, dict):
                failures.append("%s is missing in current manifest." % label)
                continue
            if expected_file.get("sha256"):
                compare_equal("%s.exists" % label, True, bool(actual_file.get("exists")), failures)
                compare_equal("%s.bytes" % label, expected_file.get("bytes"), actual_file.get("bytes"), failures)
                compare_equal("%s.sha256" % label, expected_file.get("sha256"), actual_file.get("sha256"), failures)

    return {
        "ok": not failures,
        "manifest": display_path(manifest_path.resolve()),
        "config": display_path(config_path),
        "failures": failures,
        "checked_roles": list(ROLES),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify Ascend 310B Pro routed model manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Write model_manifest.json from the current route config.")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--soc-version", default=os.environ.get("SOC_VERSION"))
    build.add_argument("--om-cache-dir", type=Path)

    verify = subparsers.add_parser("verify", help="Verify current files against model_manifest.json.")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--om-cache-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        payload = build_manifest(args.config, args.soc_version, args.om_cache_dir)
        write_json(args.output, payload)
        print(json.dumps({"ok": True, "manifest": display_path(args.output.resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify":
        result = verify_manifest(args.config, args.manifest, args.om_cache_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    raise ValueError("unsupported command: %s" % args.command)


if __name__ == "__main__":
    raise SystemExit(main())
