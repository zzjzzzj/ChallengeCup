#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_tzb_board_tests.sh \
    --before-model deployment/ascend310bpro/models/r1-four-class-detector-easy-640.pt \
    --after-model deployment/ascend310bpro/models/class-il-er500-stage06-best.pt \
    --workspace outputs/ascend310bpro_tzb_board/er500 \
    --soc-version Ascend310B4

Default test image folders:
  base test      data/testdata/base_test_r1
  increment test data/testdata/inc_test_r2

This script runs the three blind-test predictions required by attachment 1:
  base_before      before/base model on base_test_r1
  base_after       after/increment model on base_test_r1
  increment_after  after/increment model on inc_test_r2

It accepts .pt, .onnx, or .om models. .pt checkpoints are exported to cached
static batch-1 ONNX first, then ATC converts/reuses OM for NPU inference.

Options:
  --base-test PATH
  --increment-test PATH
  --before-model PATH          Default: models/r1-four-class-detector-easy-640.pt.
  --after-model PATH           ER/DER increment model, required.
  --before-classes PATH        Default: classes_base4.txt.
  --after-classes PATH         Default: classes_6.txt.
  --before-image-size N        Default: 640.
  --after-image-size N         Default: 640.
  --before-decode-mode auto|nms|raw
                               Default: auto.
  --after-decode-mode auto|nms|raw
                               Default: auto.
  --workspace PATH             Default: outputs/ascend310bpro_tzb_board.
  --export-dir PATH            Default: <workspace>/exports.
  --export-device DEVICE       Default: cpu.
  --export-opset N             Default: 12.
  --export-simplify            Pass simplify=True to Ultralytics export.
  --force-export               Rebuild cached ONNX files.
  --soc-version TEXT           Example: Ascend310B4.
  --confidence FLOAT           Default: 0.25.
  --iou FLOAT                  Default: 0.55.
  --team-id TEXT               Used for required result folder name.
  --fps-label TEXT             Used for required result folder name. Default: auto.
  --strategy-label TEXT        Written into result manifest.
  --submission-dir PATH        Exact formatted result output directory.
  --skip-format                Do not build attachment-1 TXT result folder.
  --no-save-images             Write JSON/TXT only.
  --force-convert              Rebuild cached OM files.
EOF
}

fail() {
  echo "run_tzb_board_tests.sh: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

abspath() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    realpath -m "${value}"
  else
    realpath -m "${PROJECT_ROOT}/${value}"
  fi
}

BASE_TEST="data/testdata/base_test_r1"
INCREMENT_TEST="data/testdata/inc_test_r2"
BEFORE_MODEL="${SCRIPT_DIR}/models/r1-four-class-detector-easy-640.pt"
AFTER_MODEL=""
BEFORE_CLASSES="${SCRIPT_DIR}/classes_base4.txt"
AFTER_CLASSES="${SCRIPT_DIR}/classes_6.txt"
BEFORE_IMAGE_SIZE=640
AFTER_IMAGE_SIZE=640
BEFORE_DECODE_MODE="auto"
AFTER_DECODE_MODE="auto"
WORKSPACE="outputs/ascend310bpro_tzb_board"
EXPORT_DIR=""
EXPORT_DEVICE="cpu"
EXPORT_OPSET=12
EXPORT_SIMPLIFY=0
FORCE_EXPORT=0
SOC_VERSION="${SOC_VERSION:-}"
CONFIDENCE="0.25"
IOU="0.55"
TEAM_ID="作品编号"
FPS_LABEL="auto"
STRATEGY_LABEL=""
SUBMISSION_DIR=""
FORMAT_RESULTS=1
NO_SAVE_IMAGES=0
FORCE_CONVERT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --base-test) BASE_TEST="${2:?missing value for $1}"; shift 2 ;;
    --increment-test) INCREMENT_TEST="${2:?missing value for $1}"; shift 2 ;;
    --before-model) BEFORE_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --after-model) AFTER_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --before-classes) BEFORE_CLASSES="${2:?missing value for $1}"; shift 2 ;;
    --after-classes) AFTER_CLASSES="${2:?missing value for $1}"; shift 2 ;;
    --before-image-size) BEFORE_IMAGE_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --after-image-size) AFTER_IMAGE_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --before-decode-mode) BEFORE_DECODE_MODE="${2:?missing value for $1}"; shift 2 ;;
    --after-decode-mode) AFTER_DECODE_MODE="${2:?missing value for $1}"; shift 2 ;;
    --workspace) WORKSPACE="${2:?missing value for $1}"; shift 2 ;;
    --export-dir) EXPORT_DIR="${2:?missing value for $1}"; shift 2 ;;
    --export-device) EXPORT_DEVICE="${2:?missing value for $1}"; shift 2 ;;
    --export-opset) EXPORT_OPSET="${2:?missing value for $1}"; shift 2 ;;
    --export-simplify) EXPORT_SIMPLIFY=1; shift ;;
    --force-export) FORCE_EXPORT=1; shift ;;
    --soc-version) SOC_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --confidence) CONFIDENCE="${2:?missing value for $1}"; shift 2 ;;
    --iou) IOU="${2:?missing value for $1}"; shift 2 ;;
    --team-id) TEAM_ID="${2:?missing value for $1}"; shift 2 ;;
    --fps-label) FPS_LABEL="${2:?missing value for $1}"; shift 2 ;;
    --strategy-label) STRATEGY_LABEL="${2:?missing value for $1}"; shift 2 ;;
    --submission-dir) SUBMISSION_DIR="${2:?missing value for $1}"; shift 2 ;;
    --skip-format) FORMAT_RESULTS=0; shift ;;
    --no-save-images) NO_SAVE_IMAGES=1; shift ;;
    --force-convert) FORCE_CONVERT=1; shift ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "${BEFORE_MODEL}" ]] || fail "missing --before-model"
[[ -n "${AFTER_MODEL}" ]] || fail "missing --after-model"
for mode in "${BEFORE_DECODE_MODE}" "${AFTER_DECODE_MODE}"; do
  [[ "${mode}" == "auto" || "${mode}" == "nms" || "${mode}" == "raw" ]] || fail "decode mode must be auto, nms or raw"
done

BASE_TEST="$(abspath "${BASE_TEST}")"
INCREMENT_TEST="$(abspath "${INCREMENT_TEST}")"
BEFORE_MODEL="$(abspath "${BEFORE_MODEL}")"
AFTER_MODEL="$(abspath "${AFTER_MODEL}")"
BEFORE_CLASSES="$(abspath "${BEFORE_CLASSES}")"
AFTER_CLASSES="$(abspath "${AFTER_CLASSES}")"
WORKSPACE="$(abspath "${WORKSPACE}")"
[[ -n "${EXPORT_DIR}" ]] || EXPORT_DIR="${WORKSPACE}/exports"
EXPORT_DIR="$(abspath "${EXPORT_DIR}")"
if [[ -n "${SUBMISSION_DIR}" ]]; then
  SUBMISSION_DIR="$(abspath "${SUBMISSION_DIR}")"
fi

[[ -d "${BASE_TEST}" ]] || fail "base test directory not found: ${BASE_TEST}"
[[ -d "${INCREMENT_TEST}" ]] || fail "increment test directory not found: ${INCREMENT_TEST}"
[[ -f "${BEFORE_MODEL}" ]] || fail "before model not found: ${BEFORE_MODEL}"
[[ -f "${AFTER_MODEL}" ]] || fail "after model not found: ${AFTER_MODEL}"
[[ -f "${BEFORE_CLASSES}" ]] || fail "before classes file not found: ${BEFORE_CLASSES}"
[[ -f "${AFTER_CLASSES}" ]] || fail "after classes file not found: ${AFTER_CLASSES}"

cd "${PROJECT_ROOT}"
mkdir -p "${WORKSPACE}" "${EXPORT_DIR}"

onnx_path_for_pt() {
  local model="$1"
  local image_size="$2"
  local filename
  filename="$(basename "${model}")"
  printf '%s/%s_%sx%s.onnx\n' "${EXPORT_DIR}" "${filename%.pt}" "${image_size}" "${image_size}"
}

resolve_npu_model() {
  local model="$1"
  local image_size="$2"
  case "${model}" in
    *.pt)
      local onnx_path
      onnx_path="$(onnx_path_for_pt "${model}" "${image_size}")"
      local export_cmd=(
        "${PYTHON_BIN}" "${SCRIPT_DIR}/export_yolo_pt_to_onnx.py"
        --model "${model}"
        --output "${onnx_path}"
        --image-size "${image_size}"
        --opset "${EXPORT_OPSET}"
        --device "${EXPORT_DEVICE}"
      )
      [[ "${EXPORT_SIMPLIFY}" -eq 1 ]] && export_cmd+=(--simplify)
      [[ "${FORCE_EXPORT}" -eq 1 ]] && export_cmd+=(--force)
      echo "[INFO] Export .pt to ONNX: ${model} -> ${onnx_path}" >&2
      "${export_cmd[@]}" >&2
      printf '%s\n' "${onnx_path}"
      ;;
    *.onnx|*.om)
      printf '%s\n' "${model}"
      ;;
    *)
      fail "model must be .pt, .onnx or .om: ${model}"
      ;;
  esac
}

BEFORE_MODEL="$(resolve_npu_model "${BEFORE_MODEL}" "${BEFORE_IMAGE_SIZE}")"
AFTER_MODEL="$(resolve_npu_model "${AFTER_MODEL}" "${AFTER_IMAGE_SIZE}")"

run_one() {
  local tag="$1"
  local model="$2"
  local input="$3"
  local classes="$4"
  local decode_mode="$5"
  local out_dir="${WORKSPACE}/${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/run_best_6class_npu.sh"
    --model "${model}"
    --input "${input}"
    --classes "${classes}"
    --output-dir "${out_dir}"
    --summary "${out_dir}/summary.json"
    --jsonl "${out_dir}/predictions.jsonl"
    --decode-mode "${decode_mode}"
    --confidence "${CONFIDENCE}"
    --iou "${IOU}"
  )
  if [[ -n "${SOC_VERSION}" ]]; then
    cmd+=(--soc-version "${SOC_VERSION}")
  fi
  [[ "${NO_SAVE_IMAGES}" -eq 1 ]] && cmd+=(--no-save-images)
  [[ "${FORCE_CONVERT}" -eq 1 ]] && cmd+=(--force-convert)

  echo "[INFO] Running ${tag}"
  "${cmd[@]}"
  bash "${SCRIPT_DIR}/run_agent_reports.sh" \
    --predictions "${out_dir}/predictions.jsonl" \
    --summary "${out_dir}/summary.json" \
    --output-dir "${out_dir}/agent_reports"
}

run_one "base_before" "${BEFORE_MODEL}" "${BASE_TEST}" "${BEFORE_CLASSES}" "${BEFORE_DECODE_MODE}"
run_one "base_after" "${AFTER_MODEL}" "${BASE_TEST}" "${AFTER_CLASSES}" "${AFTER_DECODE_MODE}"
run_one "increment_after" "${AFTER_MODEL}" "${INCREMENT_TEST}" "${AFTER_CLASSES}" "${AFTER_DECODE_MODE}"

bash "${SCRIPT_DIR}/run_tzb_submission.sh" \
  --skip-map \
  --npu-summary "${WORKSPACE}/increment_after/summary.json" \
  --agent-summary "${WORKSPACE}/increment_after/agent_reports/agent_summary.json" \
  --output-dir "${WORKSPACE}/submission_package"

fps_label_from_summary() {
  "${PYTHON_BIN}" -c 'import json,sys
path=sys.argv[1]
try:
    value=json.load(open(path, encoding="utf-8-sig")).get("fps")
    print(str(int(round(float(value)))) if value is not None else "待填")
except Exception:
    print("待填")
' "$1"
}

if [[ -z "${SUBMISSION_DIR}" ]]; then
  RESOLVED_FPS_LABEL="${FPS_LABEL}"
  if [[ "${RESOLVED_FPS_LABEL}" == "auto" ]]; then
    RESOLVED_FPS_LABEL="$(fps_label_from_summary "${WORKSPACE}/increment_after/summary.json")"
  fi
  SUBMISSION_DIR="${WORKSPACE}/${TEAM_ID}_FPS指标${RESOLVED_FPS_LABEL}"
fi

if [[ "${FORMAT_RESULTS}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/tzb_format_results.py" \
    --base-before-jsonl "${WORKSPACE}/base_before/predictions.jsonl" \
    --base-after-jsonl "${WORKSPACE}/base_after/predictions.jsonl" \
    --increment-after-jsonl "${WORKSPACE}/increment_after/predictions.jsonl" \
    --base-before-summary "${WORKSPACE}/base_before/summary.json" \
    --base-after-summary "${WORKSPACE}/base_after/summary.json" \
    --increment-after-summary "${WORKSPACE}/increment_after/summary.json" \
    --output-dir "${SUBMISSION_DIR}" \
    --team-id "${TEAM_ID}" \
    --strategy-label "${STRATEGY_LABEL}"
fi

cat > "${WORKSPACE}/README.md" <<EOF
# Ascend 310B TZB Board Test Outputs

This folder contains blind-test prediction and FPS evidence from Ascend 310B.
The formatted attachment-1 result folder is:

${SUBMISSION_DIR}

Raw prediction folders:

- base_before: before model on base_test_r1
- base_after: after model on base_test_r1
- increment_after: after model on inc_test_r2
- submission_package: FPS/Agent evidence package generated with --skip-map

Key files:

- base_before/predictions.jsonl
- base_after/predictions.jsonl
- increment_after/predictions.jsonl
- base_before/summary.json
- base_after/summary.json
- increment_after/summary.json
- submission_package/tzb_metrics.json
EOF

echo "[INFO] Board test outputs: ${WORKSPACE}"
echo "[INFO] Formatted TZB folder: ${SUBMISSION_DIR}"
echo "[INFO] Target detection TXT directories:"
echo "[INFO]   ${SUBMISSION_DIR}/基础模型-基础测试集推理结果/目标检测识别模块"
echo "[INFO]   ${SUBMISSION_DIR}/增量模型-基础测试集推理结果/目标检测识别模块"
echo "[INFO]   ${SUBMISSION_DIR}/增量模型-增量测试集推理结果/目标检测识别模块"
