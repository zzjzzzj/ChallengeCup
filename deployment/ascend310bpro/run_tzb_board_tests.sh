#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_tzb_board_tests.sh \
    --before-model deployment/ascend310bpro/models/before_increment_4class_640x640.onnx \
    --after-model outputs/ascend310bpro_increment/runs/batch_il_der_b500/exports/after_increment_6class_640x640.onnx \
    --workspace outputs/ascend310bpro_tzb_board \
    --soc-version Ascend310B4

Default test image folders:
  base test      data/testdata/base_test_r1
  increment test data/testdata/inc_test_r2

This script runs the three blind-test predictions required by the rules:
  base_before      before model on base_test_r1
  base_after       after model on base_test_r1
  increment_after  after model on inc_test_r2

The local data/testdata folders contain images only. This script therefore
creates predictions/FPS/report evidence; mAP/KRR/New-mAP still require official
labels or the official evaluate_tzb.py.

Options:
  --base-test PATH
  --increment-test PATH
  --before-model PATH          ONNX/OM before model for NPU inference.
  --after-model PATH           ONNX/OM after model for NPU inference.
  --before-classes PATH        Default: classes_base4.txt.
  --after-classes PATH         Default: classes_6.txt.
  --before-decode-mode nms|raw Default: raw.
  --after-decode-mode nms|raw  Default: raw.
  --workspace PATH             Default: outputs/ascend310bpro_tzb_board.
  --soc-version TEXT           Example: Ascend310B4.
  --confidence FLOAT           Default: 0.25.
  --iou FLOAT                  Default: 0.55.
  --no-save-images             Write JSON only.
  --force-convert              Rebuild cached OM files.
EOF
}

fail() {
  echo "run_tzb_board_tests.sh: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
BEFORE_MODEL=""
AFTER_MODEL=""
BEFORE_CLASSES="${SCRIPT_DIR}/classes_base4.txt"
AFTER_CLASSES="${SCRIPT_DIR}/classes_6.txt"
BEFORE_DECODE_MODE="raw"
AFTER_DECODE_MODE="raw"
WORKSPACE="outputs/ascend310bpro_tzb_board"
SOC_VERSION="${SOC_VERSION:-}"
CONFIDENCE="0.25"
IOU="0.55"
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
    --before-decode-mode) BEFORE_DECODE_MODE="${2:?missing value for $1}"; shift 2 ;;
    --after-decode-mode) AFTER_DECODE_MODE="${2:?missing value for $1}"; shift 2 ;;
    --workspace) WORKSPACE="${2:?missing value for $1}"; shift 2 ;;
    --soc-version) SOC_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --confidence) CONFIDENCE="${2:?missing value for $1}"; shift 2 ;;
    --iou) IOU="${2:?missing value for $1}"; shift 2 ;;
    --no-save-images) NO_SAVE_IMAGES=1; shift ;;
    --force-convert) FORCE_CONVERT=1; shift ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "${BEFORE_MODEL}" ]] || fail "missing --before-model"
[[ -n "${AFTER_MODEL}" ]] || fail "missing --after-model"
[[ "${BEFORE_DECODE_MODE}" == "nms" || "${BEFORE_DECODE_MODE}" == "raw" ]] || fail "--before-decode-mode must be nms or raw"
[[ "${AFTER_DECODE_MODE}" == "nms" || "${AFTER_DECODE_MODE}" == "raw" ]] || fail "--after-decode-mode must be nms or raw"

BASE_TEST="$(abspath "${BASE_TEST}")"
INCREMENT_TEST="$(abspath "${INCREMENT_TEST}")"
BEFORE_MODEL="$(abspath "${BEFORE_MODEL}")"
AFTER_MODEL="$(abspath "${AFTER_MODEL}")"
BEFORE_CLASSES="$(abspath "${BEFORE_CLASSES}")"
AFTER_CLASSES="$(abspath "${AFTER_CLASSES}")"
WORKSPACE="$(abspath "${WORKSPACE}")"

[[ -d "${BASE_TEST}" ]] || fail "base test directory not found: ${BASE_TEST}"
[[ -d "${INCREMENT_TEST}" ]] || fail "increment test directory not found: ${INCREMENT_TEST}"
[[ -f "${BEFORE_MODEL}" ]] || fail "before model not found: ${BEFORE_MODEL}"
[[ -f "${AFTER_MODEL}" ]] || fail "after model not found: ${AFTER_MODEL}"
[[ -f "${BEFORE_CLASSES}" ]] || fail "before classes file not found: ${BEFORE_CLASSES}"
[[ -f "${AFTER_CLASSES}" ]] || fail "after classes file not found: ${AFTER_CLASSES}"
case "${BEFORE_MODEL}" in *.pt) fail "before model for NPU test must be ONNX/OM, not .pt" ;; esac
case "${AFTER_MODEL}" in *.pt) fail "after model for NPU test must be ONNX/OM, not .pt" ;; esac

cd "${PROJECT_ROOT}"
mkdir -p "${WORKSPACE}"

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

cat > "${WORKSPACE}/README.md" <<EOF
# Ascend 310B TZB Board Test Outputs

This folder contains blind-test prediction and FPS evidence from Ascend 310B.
The local data/testdata folders have images only, so official mAP/KRR/New-mAP
must be computed by evaluate_tzb.py or by using fixed test labels.

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
echo "[INFO] Submit/evaluate predictions from:"
echo "[INFO]   ${WORKSPACE}/base_before/predictions.jsonl"
echo "[INFO]   ${WORKSPACE}/base_after/predictions.jsonl"
echo "[INFO]   ${WORKSPACE}/increment_after/predictions.jsonl"
