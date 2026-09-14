#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bplus/run_augment_yolo.sh \
    --data data/datasets_r1_base_train \
    --output /path/to/augmented_dataset

Options:
  --data PATH              YOLO dataset root or data.yaml. Flat roots with
                           images, labels and classes.txt in one folder are supported.
  --output PATH            Output dataset directory.
  --classes PATH           Class-name file. Default: deployment/ascend310bplus/classes_6.txt.
  --default-modality ir|sar
                           Used when image filenames do not start with ir_ or sar_.
  --split NAME             Split name under a structured dataset root. Default: train.
  --val-root PATH          Optional independent YOLO validation root.
  --val-images PATH        Optional independent validation image directory/list/file.
  --val-split NAME         Validation split name under --val-root. Default: val.
  --no-original            Generate only augmented variants.
  --reuse-existing         If output/data.yaml exists, skip augmentation.
  --rebuild-existing       Move an existing non-empty output aside and rebuild
                           a fresh dataset at the same output path.
  --python PATH            Python executable. Default: python3.
  -h, --help               Show this help.

Output:
  <output>/data.yaml
  <output>/images/train, <output>/labels/train
  <output>/images/val,   <output>/labels/val
  <output>/augmentation_manifest.csv
  <output>/augmentation_summary.json
EOF
}

fail() {
  echo "run_augment_yolo.sh: $*" >&2
  exit 2
}

is_nonempty_dir() {
  local path="$1"
  [[ -d "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH=""
OUTPUT_PATH=""
CLASSES_PATH="${SCRIPT_DIR}/classes_6.txt"
DEFAULT_MODALITY=""
SPLIT_NAME="train"
VAL_ROOT=""
VAL_IMAGES=""
VAL_SPLIT="val"
INCLUDE_ORIGINAL=1
REUSE_EXISTING=0
REBUILD_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for $1}"
      shift 2
      ;;
    --data|--data-root|--data-yaml)
      DATA_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --classes)
      CLASSES_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --default-modality)
      DEFAULT_MODALITY="${2:?missing value for $1}"
      shift 2
      ;;
    --split)
      SPLIT_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --val-root)
      VAL_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --val-images)
      VAL_IMAGES="${2:?missing value for $1}"
      shift 2
      ;;
    --val-split)
      VAL_SPLIT="${2:?missing value for $1}"
      shift 2
      ;;
    --no-original)
      INCLUDE_ORIGINAL=0
      shift
      ;;
    --reuse-existing)
      REUSE_EXISTING=1
      shift
      ;;
    --rebuild-existing)
      REBUILD_EXISTING=1
      shift
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ -n "${DATA_PATH}" ]] || fail "missing --data"
[[ -n "${OUTPUT_PATH}" ]] || fail "missing --output"
[[ -f "${CLASSES_PATH}" ]] || fail "classes file not found: ${CLASSES_PATH}"

case "${DEFAULT_MODALITY}" in
  ""|ir|sar) ;;
  *) fail "--default-modality must be ir or sar" ;;
esac
if [[ -n "${VAL_ROOT}" && -n "${VAL_IMAGES}" ]]; then
  fail "use either --val-root or --val-images, not both"
fi
if [[ "${REUSE_EXISTING}" -eq 1 && "${REBUILD_EXISTING}" -eq 1 ]]; then
  fail "use either --reuse-existing or --rebuild-existing, not both"
fi

DATA_YAML=""
AUGMENT_MODE=""
if [[ -d "${DATA_PATH}" ]]; then
  if [[ -f "${DATA_PATH}/data.yaml" ]]; then
    DATA_YAML="${DATA_PATH}/data.yaml"
    AUGMENT_MODE="data_yaml"
  else
    AUGMENT_MODE="dataset_root"
  fi
else
  DATA_YAML="${DATA_PATH}"
  [[ -f "${DATA_YAML}" ]] || fail "data.yaml not found: ${DATA_YAML}"
  AUGMENT_MODE="data_yaml"
fi

if [[ -f "${OUTPUT_PATH}/data.yaml" && "${REUSE_EXISTING}" -eq 1 ]]; then
  echo "[INFO] Reuse augmented dataset: ${OUTPUT_PATH}/data.yaml"
  exit 0
fi

if is_nonempty_dir "${OUTPUT_PATH}"; then
  if [[ "${REBUILD_EXISTING}" -eq 1 ]]; then
    output_abs="$(realpath -m "${OUTPUT_PATH}")"
    project_abs="$(realpath -m "${PROJECT_ROOT}")"
    data_abs="$(realpath -m "${DATA_PATH}")"
    case "${output_abs}" in
      ""|"/")
        fail "refuse to rebuild unsafe output path: ${OUTPUT_PATH}"
        ;;
    esac
    if [[ "${output_abs}" == "${project_abs}" || "${output_abs}" == "${data_abs}" ]]; then
      fail "refuse to rebuild output path that points at project/data root: ${output_abs}"
    fi
    backup_path="${OUTPUT_PATH}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] Existing augmented output will be moved aside:"
    echo "[INFO]   old: ${OUTPUT_PATH}"
    echo "[INFO]   bak: ${backup_path}"
    mv -- "${OUTPUT_PATH}" "${backup_path}"
  else
    fail "output directory is not empty: ${OUTPUT_PATH}
Pass --reuse-existing to use it, --rebuild-existing to regenerate it in place, or choose a new --output path."
  fi
fi

cd "${PROJECT_ROOT}"

echo "[INFO] Augment YOLO dataset"
echo "[INFO] Data  : ${DATA_PATH}"
echo "[INFO] Output: ${OUTPUT_PATH}"
echo "[INFO] Classes: ${CLASSES_PATH}"
if [[ "${AUGMENT_MODE}" == "data_yaml" ]]; then
  if [[ -n "${VAL_ROOT}" || -n "${VAL_IMAGES}" ]]; then
    fail "--val-root/--val-images are only used for flat dataset roots without data.yaml"
  fi
  cmd=(
    "${PYTHON_BIN}" train.py augment-yolo
    --data "${DATA_YAML}"
    --output "${OUTPUT_PATH}"
  )

  if [[ -n "${DEFAULT_MODALITY}" ]]; then
    cmd+=(--default-modality "${DEFAULT_MODALITY}")
  fi

  if [[ "${INCLUDE_ORIGINAL}" -eq 0 ]]; then
    cmd+=(--no-original)
  fi
else
  cmd=(
    "${PYTHON_BIN}" deployment/ascend310b/augment_selected_yolo.py
    --dataset-root "${DATA_PATH}"
    --split "${SPLIT_NAME}"
    --output "${OUTPUT_PATH}"
    --classes "${CLASSES_PATH}"
  )

  if [[ "${INCLUDE_ORIGINAL}" -eq 1 ]]; then
    cmd+=(--include-original)
  fi

  if [[ -n "${DEFAULT_MODALITY}" ]]; then
    cmd+=(--default-modality "${DEFAULT_MODALITY}")
  fi
  if [[ -n "${VAL_ROOT}" ]]; then
    cmd+=(--val-root "${VAL_ROOT}" --val-split "${VAL_SPLIT}")
  fi
  if [[ -n "${VAL_IMAGES}" ]]; then
    cmd+=(--val-images "${VAL_IMAGES}")
  fi
fi

"${cmd[@]}"
echo "[INFO] Augmented data.yaml: ${OUTPUT_PATH}/data.yaml"
