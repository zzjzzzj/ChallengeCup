#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_release_weight_tzb_tests.sh \
    --soc-version Ascend310B4 \
    --team-id 作品编号

This runs attachment-1 board tests for the current release weight set:
  before/base: r1-four-class-detector-easy-640.pt
  after ER500 : class-il-er500-stage06-best.pt
  after DER500: class-il-der500-stage06-best.pt

Outputs:
  <workspace>/er500/...
  <workspace>/der500/...

Options:
  --base-test PATH             Default: data/testdata/base_test_r1.
  --increment-test PATH        Default: data/testdata/inc_test_r2.
  --base-model PATH
  --er-model PATH
  --der-model PATH
  --workspace PATH             Default: outputs/ascend310bpro_release_tzb.
  --soc-version TEXT           Example: Ascend310B4.
  --team-id TEXT               Default: 作品编号.
  --fps-label-er TEXT          Default: auto.
  --fps-label-der TEXT         Default: auto.
  --confidence FLOAT           Default: 0.25.
  --iou FLOAT                  Default: 0.55.
  --export-device DEVICE       Default: cpu.
  --force-export
  --force-convert
  --no-save-images
EOF
}

fail() {
  echo "run_release_weight_tzb_tests.sh: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_TEST="data/testdata/base_test_r1"
INCREMENT_TEST="data/testdata/inc_test_r2"
BASE_MODEL="${SCRIPT_DIR}/models/r1-four-class-detector-easy-640.pt"
ER_MODEL="${SCRIPT_DIR}/models/class-il-er500-stage06-best.pt"
DER_MODEL="${SCRIPT_DIR}/models/class-il-der500-stage06-best.pt"
WORKSPACE="outputs/ascend310bpro_release_tzb"
SOC_VERSION="${SOC_VERSION:-}"
TEAM_ID="作品编号"
FPS_LABEL_ER="auto"
FPS_LABEL_DER="auto"
CONFIDENCE="0.25"
IOU="0.55"
EXPORT_DEVICE="cpu"
FORCE_EXPORT=0
FORCE_CONVERT=0
NO_SAVE_IMAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --base-test) BASE_TEST="${2:?missing value for $1}"; shift 2 ;;
    --increment-test) INCREMENT_TEST="${2:?missing value for $1}"; shift 2 ;;
    --base-model) BASE_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --er-model) ER_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --der-model) DER_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --workspace) WORKSPACE="${2:?missing value for $1}"; shift 2 ;;
    --soc-version) SOC_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --team-id) TEAM_ID="${2:?missing value for $1}"; shift 2 ;;
    --fps-label-er) FPS_LABEL_ER="${2:?missing value for $1}"; shift 2 ;;
    --fps-label-der) FPS_LABEL_DER="${2:?missing value for $1}"; shift 2 ;;
    --confidence) CONFIDENCE="${2:?missing value for $1}"; shift 2 ;;
    --iou) IOU="${2:?missing value for $1}"; shift 2 ;;
    --export-device) EXPORT_DEVICE="${2:?missing value for $1}"; shift 2 ;;
    --force-export) FORCE_EXPORT=1; shift ;;
    --force-convert) FORCE_CONVERT=1; shift ;;
    --no-save-images) NO_SAVE_IMAGES=1; shift ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "${SOC_VERSION}" ]] || fail "missing --soc-version Ascend310B4"

run_strategy() {
  local label="$1"
  local after_model="$2"
  local fps_label="$3"
  local strategy_workspace="${WORKSPACE}/${label}"
  local cmd=(
    bash "${SCRIPT_DIR}/run_tzb_board_tests.sh"
    --base-test "${BASE_TEST}"
    --increment-test "${INCREMENT_TEST}"
    --before-model "${BASE_MODEL}"
    --after-model "${after_model}"
    --workspace "${strategy_workspace}"
    --export-dir "${WORKSPACE}/exports"
    --export-device "${EXPORT_DEVICE}"
    --soc-version "${SOC_VERSION}"
    --confidence "${CONFIDENCE}"
    --iou "${IOU}"
    --team-id "${TEAM_ID}"
    --fps-label "${fps_label}"
    --strategy-label "${label}"
  )
  [[ "${FORCE_EXPORT}" -eq 1 ]] && cmd+=(--force-export)
  [[ "${FORCE_CONVERT}" -eq 1 ]] && cmd+=(--force-convert)
  [[ "${NO_SAVE_IMAGES}" -eq 1 ]] && cmd+=(--no-save-images)
  echo "[INFO] ===== Running ${label} ====="
  "${cmd[@]}"
}

cd "${PROJECT_ROOT}"
run_strategy "er500" "${ER_MODEL}" "${FPS_LABEL_ER}"
run_strategy "der500" "${DER_MODEL}" "${FPS_LABEL_DER}"

echo "[INFO] Release weight TZB tests finished: ${WORKSPACE}"
echo "[INFO] ER500 formatted folder is under: ${WORKSPACE}/er500"
echo "[INFO] DER500 formatted folder is under: ${WORKSPACE}/der500"
