#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

default_best_model_path() {
  local candidate
  for candidate in \
    "${SCRIPT_DIR}/models/best.onnx" \
    "${SCRIPT_DIR}/../best.onnx" \
    "${PROJECT_ROOT}/models/best.onnx"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${SCRIPT_DIR}/models/best.onnx"
}

usage() {
  cat <<'EOF'
Usage:
  check_models.sh [routed model args]
  check_models.sh --single [--model PATH] [--classes PATH]

Default mode checks the report-aligned routed deployment and verifies the
model manifest when it exists.

Use --single for the compact six-class detector. In single mode the script
checks, in order:
  deployment/ascend310bpro/models/best.onnx
  deployment/best.onnx
  models/best.onnx

Options:
  --config PATH
      Routed YAML/JSON config. Default: deployment/ascend310bpro/route_config.yaml.
  --manifest PATH
      Manifest path. Default: deployment/ascend310bpro/model_manifest.json.
  --skip-manifest
      Do not verify the manifest.
  --write-manifest
      Rebuild the manifest after checking configured model paths.
  --single
      Check the auto-detected best.onnx single-detector baseline.
  --model PATH, --single-model PATH, --best-model PATH
      Single six-class ONNX/OM model. Implies --single.
  --classes PATH
      Six-class names file. Default: classes_6.txt.
  --routed
      Explicitly check the scene/easy/hard routed deployment.
  -h, --help
      Show this help.
EOF
}

fail() {
  echo "check_models.sh: $*" >&2
  exit 1
}

resolve_existing_path() {
  local input="$1"

  if [[ "${input}" = /* ]]; then
    [[ -f "${input}" ]] && printf '%s\n' "${input}" && return 0
  else
    [[ -f "${input}" ]] && printf '%s\n' "${input}" && return 0
    [[ -f "${SCRIPT_DIR}/${input}" ]] && printf '%s\n' "${SCRIPT_DIR}/${input}" && return 0
    [[ -f "${PROJECT_ROOT}/${input}" ]] && printf '%s\n' "${PROJECT_ROOT}/${input}" && return 0
  fi

  return 1
}

MODE="routed"
CONFIG_PATH="${SCRIPT_DIR}/route_config.yaml"
MANIFEST_PATH="${SCRIPT_DIR}/model_manifest.json"
SKIP_MANIFEST=0
WRITE_MANIFEST=0
MODEL_PATH=""
CLASSES_PATH="${SCRIPT_DIR}/classes_6.txt"
SOC_VERSION_VALUE="${SOC_VERSION:-}"
OM_CACHE_DIR=""
ROUTED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --routed)
      MODE="routed"
      shift
      ;;
    --single)
      MODE="single"
      MODEL_PATH="$(default_best_model_path)"
      shift
      ;;
    --config)
      CONFIG_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --skip-manifest)
      SKIP_MANIFEST=1
      shift
      ;;
    --write-manifest)
      WRITE_MANIFEST=1
      shift
      ;;
    --soc-version)
      SOC_VERSION_VALUE="${2:?missing value for $1}"
      ROUTED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --om-cache-dir)
      OM_CACHE_DIR="${2:?missing value for $1}"
      ROUTED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --model|--single-model|--best-model)
      MODE="single"
      MODEL_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --classes)
      CLASSES_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ROUTED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${MODE}" == "routed" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/routed_infer_npu.py" \
    --config "${CONFIG_PATH}" \
    --check-models \
    "${ROUTED_ARGS[@]}"
  if [[ "${WRITE_MANIFEST}" -eq 1 ]]; then
    manifest_cmd=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/model_manifest.py" build
      --config "${CONFIG_PATH}"
      --output "${MANIFEST_PATH}"
    )
    if [[ -n "${SOC_VERSION_VALUE}" ]]; then
      manifest_cmd+=(--soc-version "${SOC_VERSION_VALUE}")
    fi
    if [[ -n "${OM_CACHE_DIR}" ]]; then
      manifest_cmd+=(--om-cache-dir "${OM_CACHE_DIR}")
    fi
    "${manifest_cmd[@]}"
  elif [[ "${SKIP_MANIFEST}" -eq 0 && -f "${MANIFEST_PATH}" ]]; then
    manifest_cmd=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/model_manifest.py" verify
      --config "${CONFIG_PATH}"
      --manifest "${MANIFEST_PATH}"
    )
    if [[ -n "${OM_CACHE_DIR}" ]]; then
      manifest_cmd+=(--om-cache-dir "${OM_CACHE_DIR}")
    fi
    "${manifest_cmd[@]}"
  fi
  exit $?
fi

[[ -n "${MODEL_PATH}" ]] || MODEL_PATH="$(default_best_model_path)"
MODEL_RESOLVED="$(resolve_existing_path "${MODEL_PATH}")" || fail "model not found: ${MODEL_PATH}"
CLASSES_RESOLVED="$(resolve_existing_path "${CLASSES_PATH}")" || fail "classes file not found: ${CLASSES_PATH}"

CLASS_COUNT="$(grep -cve '^[[:space:]]*$' "${CLASSES_RESOLVED}")"
[[ "${CLASS_COUNT}" == "6" ]] || fail "expected 6 classes in ${CLASSES_RESOLVED}, got ${CLASS_COUNT}"

echo "[INFO] Single six-class model: ${MODEL_RESOLVED}"
echo "[INFO] Classes file          : ${CLASSES_RESOLVED}"
echo "[INFO] Class count           : ${CLASS_COUNT}"
