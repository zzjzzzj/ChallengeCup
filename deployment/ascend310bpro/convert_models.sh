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
  convert_models.sh [--soc-version Ascend310B4] [routed model args]
  convert_models.sh --single [--model PATH] [--soc-version Ascend310B4] [single model args]

Default mode converts/reuses the report-aligned routed deployment:
  scene router 224 -> easy detector 640 / hard detector 960

Use --single for the compact six-class detector. In single mode the script
checks, in order:
  deployment/ascend310bpro/models/best.onnx
  deployment/best.onnx
  models/best.onnx

Options:
  --config PATH
      Routed YAML/JSON config. Default: deployment/ascend310bpro/route_config.yaml.
  --single
      Use the auto-detected best.onnx single-detector baseline.
  --model PATH, --single-model PATH, --best-model PATH
      Single six-class ONNX/OM model. Implies --single.
  --routed
      Explicitly use the scene/easy/hard routed deployment.
  -h, --help
      Show this help.

All other options are passed to the selected Python runtime.
EOF
}

MODE="routed"
CONFIG_PATH="${SCRIPT_DIR}/route_config.yaml"
MODEL_PATH=""
ARGS=()

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
    --model|--single-model|--best-model)
      MODE="single"
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
    --config "${CONFIG_PATH}" \
    --convert-only \
    "${ARGS[@]}"
else
  [[ -n "${MODEL_PATH}" ]] || MODEL_PATH="$(default_best_model_path)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_best_6class_npu.py" \
    --convert-only \
    --model "${MODEL_PATH}" \
    --classes "${SCRIPT_DIR}/classes_6.txt" \
    "${ARGS[@]}"
fi
