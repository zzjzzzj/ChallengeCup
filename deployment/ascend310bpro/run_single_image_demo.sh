#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_single_image_demo.sh \
    --image path/to/image.jpg \
    --strategy er500 \
    --soc-version Ascend310B4

This is a small demo wrapper: one input image in, one annotated result image out.

Options:
  --image PATH                Input image, required.
  --strategy er500|der500|base
                              Default: er500.
  --model PATH                Override model path. Accepts .pt, .onnx, or .om.
  --classes PATH              Override classes file.
  --output-dir PATH           Default: outputs/ascend310bpro_demo/<strategy>.
  --export-dir PATH           Default: outputs/ascend310bpro_release_tzb/exports.
  --soc-version TEXT          Example: Ascend310B4. Can also use SOC_VERSION env.
  --confidence FLOAT          Default: 0.25.
  --iou FLOAT                 Default: 0.55.
  --export-device DEVICE      Default: cpu.
  --force-export              Rebuild cached ONNX if model is .pt.
  --force-convert             Rebuild cached OM.
EOF
}

fail() {
  echo "run_single_image_demo.sh: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

abspath() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    realpath -m "${value}"
  else
    realpath -m "${PROJECT_ROOT}/${value}"
  fi
}

IMAGE=""
STRATEGY="er500"
MODEL=""
CLASSES=""
OUTPUT_DIR=""
EXPORT_DIR="outputs/ascend310bpro_release_tzb/exports"
SOC_VERSION="${SOC_VERSION:-}"
CONFIDENCE="0.25"
IOU="0.55"
EXPORT_DEVICE="cpu"
FORCE_EXPORT=0
FORCE_CONVERT=0
IMAGE_SIZE=640

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --image) IMAGE="${2:?missing value for $1}"; shift 2 ;;
    --strategy) STRATEGY="${2:?missing value for $1}"; shift 2 ;;
    --model) MODEL="${2:?missing value for $1}"; shift 2 ;;
    --classes) CLASSES="${2:?missing value for $1}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing value for $1}"; shift 2 ;;
    --export-dir) EXPORT_DIR="${2:?missing value for $1}"; shift 2 ;;
    --soc-version) SOC_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --confidence) CONFIDENCE="${2:?missing value for $1}"; shift 2 ;;
    --iou) IOU="${2:?missing value for $1}"; shift 2 ;;
    --export-device) EXPORT_DEVICE="${2:?missing value for $1}"; shift 2 ;;
    --force-export) FORCE_EXPORT=1; shift ;;
    --force-convert) FORCE_CONVERT=1; shift ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "${IMAGE}" ]] || fail "missing --image"
[[ -n "${SOC_VERSION}" ]] || fail "missing --soc-version Ascend310B4"

case "${STRATEGY}" in
  er500)
    DEFAULT_MODEL="${SCRIPT_DIR}/models/class-il-er500-stage06-best.pt"
    DEFAULT_CLASSES="${SCRIPT_DIR}/classes_6.txt"
    ;;
  der500)
    DEFAULT_MODEL="${SCRIPT_DIR}/models/class-il-der500-stage06-best.pt"
    DEFAULT_CLASSES="${SCRIPT_DIR}/classes_6.txt"
    ;;
  base)
    DEFAULT_MODEL="${SCRIPT_DIR}/models/r1-four-class-detector-easy-640.pt"
    DEFAULT_CLASSES="${SCRIPT_DIR}/classes_base4.txt"
    ;;
  *)
    fail "--strategy must be er500, der500, or base"
    ;;
esac

[[ -n "${MODEL}" ]] || MODEL="${DEFAULT_MODEL}"
[[ -n "${CLASSES}" ]] || CLASSES="${DEFAULT_CLASSES}"
[[ -n "${OUTPUT_DIR}" ]] || OUTPUT_DIR="outputs/ascend310bpro_demo/${STRATEGY}"

cd "${PROJECT_ROOT}"

IMAGE="$(abspath "${IMAGE}")"
MODEL="$(abspath "${MODEL}")"
CLASSES="$(abspath "${CLASSES}")"
OUTPUT_DIR="$(abspath "${OUTPUT_DIR}")"
EXPORT_DIR="$(abspath "${EXPORT_DIR}")"

[[ -f "${IMAGE}" ]] || fail "image not found: ${IMAGE}"
[[ -f "${MODEL}" ]] || fail "model not found: ${MODEL}"
[[ -f "${CLASSES}" ]] || fail "classes file not found: ${CLASSES}"
mkdir -p "${OUTPUT_DIR}" "${EXPORT_DIR}"

onnx_path_for_pt() {
  local model="$1"
  local filename
  filename="$(basename "${model}")"
  printf '%s/%s_%sx%s.onnx\n' "${EXPORT_DIR}" "${filename%.pt}" "${IMAGE_SIZE}" "${IMAGE_SIZE}"
}

resolve_npu_model() {
  local model="$1"
  case "${model}" in
    *.pt)
      local onnx_path
      onnx_path="$(onnx_path_for_pt "${model}")"
      local export_cmd=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/export_yolo_pt_to_onnx.py"
        --model "${model}"
        --output "${onnx_path}"
        --image-size "${IMAGE_SIZE}"
        --opset 12
        --device "${EXPORT_DEVICE}"
      )
      [[ "${FORCE_EXPORT}" -eq 1 ]] && export_cmd+=(--force)
      echo "[INFO] Export or reuse ONNX: ${model} -> ${onnx_path}" >&2
      "${export_cmd[@]}" >&2
      printf '%s\n' "${onnx_path}"
      ;;
    *.onnx|*.om)
      printf '%s\n' "${model}"
      ;;
    *)
      fail "model must be .pt, .onnx, or .om: ${model}"
      ;;
  esac
}

NPU_MODEL="$(resolve_npu_model "${MODEL}")"

cmd=(
  bash "${SCRIPT_DIR}/run_best_6class_npu.sh"
  --model "${NPU_MODEL}"
  --input "${IMAGE}"
  --classes "${CLASSES}"
  --output-dir "${OUTPUT_DIR}"
  --summary "${OUTPUT_DIR}/summary.json"
  --jsonl "${OUTPUT_DIR}/predictions.jsonl"
  --om-cache-dir "${EXPORT_DIR}"
  --soc-version "${SOC_VERSION}"
  --confidence "${CONFIDENCE}"
  --iou "${IOU}"
  --decode-mode auto
)
[[ "${FORCE_CONVERT}" -eq 1 ]] && cmd+=(--force-convert)

echo "[INFO] Demo image : ${IMAGE}"
echo "[INFO] Strategy   : ${STRATEGY}"
echo "[INFO] Model      : ${NPU_MODEL}"
echo "[INFO] Output dir : ${OUTPUT_DIR}"
"${cmd[@]}"

"${PYTHON_BIN}" -c 'import json, sys
from pathlib import Path
jsonl = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
row = rows[0] if rows else {}
image = Path(row.get("image", "demo.jpg"))
dets = row.get("detections") or []
print("[INFO] Detection count: %d" % len(dets))
for idx, det in enumerate(dets[:20], 1):
    print("[INFO]   #%d %s %.4f box=%s" % (idx, det.get("class_name"), float(det.get("score", 0.0)), det.get("box")))
print("[INFO] Annotated image: %s" % (out_dir / "images" / image.with_suffix(".jpg").name))
print("[INFO] Prediction JSONL: %s" % jsonl)
print("[INFO] Summary JSON    : %s" % (out_dir / "summary.json"))
' "${OUTPUT_DIR}/predictions.jsonl" "${OUTPUT_DIR}"
