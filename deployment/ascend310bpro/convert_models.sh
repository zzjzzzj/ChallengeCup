#!/usr/bin/env bash
set -euo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

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
  convert_models.sh [--model PATH] [--soc-version Ascend310B4] [extra args]
  convert_models.sh --routed [routed model args]

Default mode converts/reuses the current single six-class detector. The script
checks, in order:
  deployment/ascend310bpro/models/best.onnx
  deployment/best.onnx
  models/best.onnx

Options:
  --model PATH, --single-model PATH, --best-model PATH
      Single six-class ONNX/OM model. Default: auto-detected best.onnx.
  --routed
      Use the older scene/easy/hard routed deployment from config.json.
  -h, --help
      Show this help.

All other options are passed to the selected Python runtime.
EOF
}

MODE="single"
MODEL_PATH="$(default_best_model_path)"
ARGS=()

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${MODE}" == "routed" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/routed_infer_npu.py" \
    --config "${SCRIPT_DIR}/config.json" \
    --convert-only \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_best_6class_npu.py" \
    --convert-only \
    --model "${MODEL_PATH}" \
    --classes "${SCRIPT_DIR}/classes_6.txt" \
    "${ARGS[@]}"
fi
