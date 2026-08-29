#!/usr/bin/env bash
set -euo pipefail

ONNX_MODEL="${1:-detector_yolov8n_bs1.onnx}"
OUTPUT_PREFIX="${2:-detector_yolov8n_960_bs1}"
IMAGE_SIZE="${IMAGE_SIZE:-960}"
IMAGE_WIDTH="${IMAGE_WIDTH:-${IMAGE_SIZE}}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-${IMAGE_SIZE}}"
INPUT_NAME="${INPUT_NAME:-images}"
SOC_VERSION="${SOC_VERSION:-}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

if ! command -v atc >/dev/null 2>&1; then
  echo "atc was not found. Install CANN toolkit and source set_env.sh first." >&2
  exit 1
fi

if [ -z "${SOC_VERSION}" ]; then
  echo "SOC_VERSION is not set. Check the value with: npu-smi info; atc --list_soc_version" >&2
  echo "Example: export SOC_VERSION=Ascend310B4" >&2
  exit 2
fi

atc \
  --model="${ONNX_MODEL}" \
  --framework=5 \
  --output="${OUTPUT_PREFIX}" \
  --input_format=NCHW \
  --input_shape="${INPUT_NAME}:1,3,${IMAGE_HEIGHT},${IMAGE_WIDTH}" \
  --precision_mode=allow_fp32_to_fp16 \
  --soc_version="${SOC_VERSION}"

echo "Created ${OUTPUT_PREFIX}.om"
