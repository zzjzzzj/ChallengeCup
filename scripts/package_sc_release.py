"""Package selected SC model weights and sanitized run results for a release.

The source tree is intentionally the local, Git-ignored private artifact root.
Only explicitly enumerated weights and result files are exported. Dataset images,
labels, split manifests, and raw predictions are never traversed or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RELEASE_TAG = "sc-models-2026-09-15"


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    source: str
    output_name: str
    protocol: str
    status: str


MODEL_ARTIFACTS = (
    ModelArtifact(
        "class-il-generic-initializer",
        "yolo26n.pt",
        "class-il-generic-yolo26n-initializer.pt",
        "generic initializer used by the formal six-stage Class-IL experiments",
        "generic pretrained checkpoint; not the R1 four-class base detector",
    ),
    ModelArtifact(
        "r1-four-class-base",
        "models/r1_base_20260729/detector_easy_640.pt",
        "r1-four-class-detector-easy-640.pt",
        "R1 four-class base detector",
        "selected base checkpoint",
    ),
    ModelArtifact(
        "r1-to-r2-replay200",
        "runs/continual_compare_20260825/replay200_seed42_e30/weights/best.pt",
        "r1-to-r2-replay200-best.pt",
        "four-class to six-class replay, 200 samples, 30 epochs",
        "recommended six-class candidate for the R1-to-R2 experiment",
    ),
    ModelArtifact(
        "class-il-er200",
        "runs/class_il_er_b200_seed42_e30_tvt/stage_06_armored_vehicle/weights/best.pt",
        "class-il-er200-stage06-best.pt",
        "six-stage Class-IL, ER, buffer 200",
        "formal single-seed experiment",
    ),
    ModelArtifact(
        "class-il-er500",
        "runs/class_il_er_b500_seed42_e30_tvt/stage_06_armored_vehicle/weights/best.pt",
        "class-il-er500-stage06-best.pt",
        "six-stage Class-IL, ER, buffer 500",
        "best final accuracy in the formal single-seed comparison",
    ),
    ModelArtifact(
        "class-il-der200",
        "runs/class_il_der_b200_seed42_e30_tvt/stage_06_armored_vehicle/weights/best.pt",
        "class-il-der200-stage06-best.pt",
        "six-stage Class-IL, DER, buffer 200",
        "formal single-seed experiment",
    ),
    ModelArtifact(
        "class-il-der500",
        "runs/class_il_der_b500_seed42_e30_tvt/stage_06_armored_vehicle/weights/best.pt",
        "class-il-der500-stage06-best.pt",
        "six-stage Class-IL, DER, buffer 500",
        "best old-class retention in the formal single-seed comparison",
    ),
    ModelArtifact(
        "four-to-six-der200-sparse-moe-smoke",
        "smoke_four_to_six_20260831/runs/der_sparse_two_batch_final/batch_02/weights/best.pt",
        "four-to-six-der200-sparse-moe-smoke-best.pt",
        "four-class to six-class, two-batch DER-200 with Sparse-MoE",
        "one-epoch smoke checkpoint; not a formal accuracy result",
    ),
)


RESULT_FILES = (
    "runs/continual_compare_20260825/comparison_report.md",
    "runs/continual_compare_20260825/validation_comparison_101point.json",
    "runs/continual_compare_20260825/replay200_seed42_e30/continual_training_summary.json",
    "runs/continual_compare_20260825/replay200_seed42_e30/results.csv",
    "runs/continual_compare_20260825/increment_only_seed42_e30/results.csv",
    "runs/class_il_er_b200_seed42_e30_tvt/class_incremental_training_summary.json",
    "runs/class_il_er_b500_seed42_e30_tvt/class_incremental_training_summary.json",
    "runs/class_il_der_b200_seed42_e30_tvt/class_incremental_training_summary.json",
    "runs/class_il_der_b500_seed42_e30_tvt/class_incremental_training_summary.json",
    "smoke_four_to_six_20260831/runs/der_sparse_two_batch_final/batch_incremental_training_summary.json",
)


CLASS_IL_RUNS = (
    "class_il_er_b200_seed42_e30_tvt",
    "class_il_er_b500_seed42_e30_tvt",
    "class_il_der_b200_seed42_e30_tvt",
    "class_il_der_b500_seed42_e30_tvt",
)


WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)^[a-z]:[\\/]")
WINDOWS_ABSOLUTE_PATH_ANYWHERE = re.compile(r"(?i)[a-z]:[\\/]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_relative(value: str, private_root: Path) -> str | None:
    if not WINDOWS_ABSOLUTE_PATH.match(value):
        return None
    try:
        return str(Path(value).resolve().relative_to(private_root)).replace("\\", "/")
    except (OSError, ValueError):
        return None


def sanitize_value(value: Any, private_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item, private_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, private_root) for item in value]
    if isinstance(value, str) and WINDOWS_ABSOLUTE_PATH.match(value):
        relative = _private_relative(value, private_root)
        if relative is not None:
            return f"<LOCAL_PRIVATE_ROOT>/{relative}"
        return f"<REDACTED_LOCAL_PATH>/{Path(value).name}"
    return value


def sanitize_text(text: str, private_root: Path) -> str:
    variants = {
        str(private_root),
        str(private_root).replace("\\", "/"),
    }
    for value in sorted(variants, key=len, reverse=True):
        text = text.replace(value, "<LOCAL_PRIVATE_ROOT>")
    return text


def copy_sanitized(source: Path, target: Path, private_root: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        sanitized = sanitize_value(payload, private_root)
        target.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if suffix in {".md", ".txt"}:
        target.write_text(
            sanitize_text(source.read_text(encoding="utf-8-sig"), private_root),
            encoding="utf-8",
        )
        return
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
            rows = list(csv.reader(source_handle))
        with target.open("w", encoding="utf-8", newline="") as target_handle:
            csv.writer(target_handle, lineterminator="\n").writerows(rows)
        return
    raise ValueError(f"unsupported result file type: {source}")


def iter_stage_results(private_root: Path) -> Iterable[tuple[Path, Path]]:
    runs_root = private_root / "runs"
    for run_name in CLASS_IL_RUNS:
        run_root = runs_root / run_name
        for source in sorted(run_root.glob("stage_*/results.csv")):
            relative = Path("class_il_epochs") / run_name / source.parent.name / source.name
            yield source, relative


def validate_sources(private_root: Path) -> None:
    missing = [private_root / item.source for item in MODEL_ARTIFACTS if not (private_root / item.source).is_file()]
    missing.extend(private_root / item for item in RESULT_FILES if not (private_root / item).is_file())
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"required release sources are missing:\n{formatted}")


def package(private_root: Path, output: Path) -> dict[str, Any]:
    private_root = private_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    validate_sources(private_root)
    output.mkdir(parents=True, exist_ok=True)

    manifest_models: list[dict[str, Any]] = []
    for item in MODEL_ARTIFACTS:
        source = private_root / item.source
        target = output / item.output_name
        shutil.copy2(source, target)
        manifest_models.append(
            {
                "id": item.artifact_id,
                "asset": item.output_name,
                "protocol": item.protocol,
                "status": item.status,
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    results_root = output / "results"
    result_manifest: list[str] = []
    for relative in RESULT_FILES:
        source = private_root / relative
        target_relative = Path(relative)
        target = results_root / target_relative
        copy_sanitized(source, target, private_root)
        result_manifest.append(target_relative.as_posix())
    for source, target_relative in iter_stage_results(private_root):
        copy_sanitized(source, results_root / target_relative, private_root)
        result_manifest.append(target_relative.as_posix())

    leaked: list[str] = []
    for path in results_root.rglob("*"):
        if path.is_file() and WINDOWS_ABSOLUTE_PATH_ANYWHERE.search(
            path.read_text(encoding="utf-8", errors="ignore")
        ):
            leaked.append(str(path))
    if leaked:
        raise ValueError("absolute local paths remain in sanitized results: " + ", ".join(leaked))

    result_zip = output / "sc-run-results-sanitized.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    shutil.rmtree(results_root)

    manifest: dict[str, Any] = {
        "release_tag": RELEASE_TAG,
        "privacy": {
            "datasets_included": False,
            "labels_included": False,
            "raw_images_included": False,
            "absolute_paths_redacted": True,
        },
        "models": manifest_models,
        "results": {
            "asset": result_zip.name,
            "files": sorted(result_manifest),
            "size_bytes": result_zip.stat().st_size,
            "sha256": sha256(result_zip),
        },
    }
    manifest_path = output / "sc-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_paths = [output / item.output_name for item in MODEL_ARTIFACTS]
    checksum_paths.extend((result_zip, manifest_path))
    checksum_file = output / "SHA256SUMS.txt"
    checksum_file.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = package(args.private_root, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
