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
  check_models.sh [--model PATH] [--classes PATH]
  check_models.sh --routed [routed model args]

Default mode checks the current single six-class detector. The script checks,
in order:
  deployment/ascend310bpro/models/best.onnx
  deployment/best.onnx
  models/best.onnx

Options:
  --model PATH, --single-model PATH, --best-model PATH
      Single six-class ONNX/OM model. Default: auto-detected best.onnx.
  --classes PATH
      Six-class names file. Default: classes_6.txt.
  --routed
      Check the older scene/easy/hard routed deployment from config.json.
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

MODE="single"
MODEL_PATH="$(default_best_model_path)"
CLASSES_PATH="${SCRIPT_DIR}/classes_6.txt"
ROUTED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --routed)
      MODE="routed"
      shift
      ;;
    --model|--single-model|--best-model)
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
    --config "${SCRIPT_DIR}/config.json" \
    --check-models \
    "${ROUTED_ARGS[@]}"
  exit $?
fi

MODEL_RESOLVED="$(resolve_existing_path "${MODEL_PATH}")" || fail "model not found: ${MODEL_PATH}"
CLASSES_RESOLVED="$(resolve_existing_path "${CLASSES_PATH}")" || fail "classes file not found: ${CLASSES_PATH}"

CLASS_COUNT="$(grep -cve '^[[:space:]]*$' "${CLASSES_RESOLVED}")"
[[ "${CLASS_COUNT}" == "6" ]] || fail "expected 6 classes in ${CLASSES_RESOLVED}, got ${CLASS_COUNT}"

echo "[INFO] Single six-class model: ${MODEL_RESOLVED}"
echo "[INFO] Classes file          : ${CLASSES_RESOLVED}"
echo "[INFO] Class count           : ${CLASS_COUNT}"
