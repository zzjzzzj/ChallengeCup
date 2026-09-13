#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bplus/run_class_il_training.sh [options]

Common examples:
  bash deployment/ascend310bplus/run_class_il_training.sh \
    --private-root "$HOME/Desktop/workspace/ChallengeCup/private_data" \
    --method both

  bash deployment/ascend310bplus/run_class_il_training.sh \
    --data /path/to/yolo_r1_r2inc_augmented_full_tvt_seed42 \
    --prepared /path/to/class_il_prepared_sparse_moe_seed42 \
    --initial-model /path/to/yolo26n.pt \
    --output-root /path/to/runs \
    --method der

Required paths can be provided directly, or derived from --private-root:
  --private-root PATH         Base private data folder.
  --data PATH                 YOLO dataset root or data.yaml.
  --prepared PATH             Prepared Class-IL output/reuse directory.
  --initial-model PATH        Initial YOLO .pt or .yaml model.
  --output-root PATH          Parent folder for ER/DER run outputs.
  --der-output PATH           Explicit DER output directory.
  --er-output PATH            Explicit ER output directory.

Main controls:
  --method der|er|both        Training method to run. Default: both.
  --skip-prepare              Reuse an existing prepared directory.
  --prepare-only              Only run prepare-class-il.
  --train-only                Same as --skip-prepare.
  --device DEVICE             Default: cpu. Use npu:0 only with a working torch_npu stack.
  --batch-size N              Default: 2, safer on the board.
  --workers N                 Default: 0, safer on the board.
  --epochs N                  Default: 30.
  --smoke-test                Run 1 epoch and stop after stage 1.

Advanced controls:
  --python PATH               Python executable. Default: python3.
  --data-name NAME            Derived dataset folder name under --private-root.
  --prepared-name NAME        Derived prepared folder name under --private-root.
  --model-name NAME           Derived initial model filename under --private-root.
  --prepare-buffer-size N     Repeatable. Default: 200 and 500.
  --buffer-size N             Training replay buffer. Default: 500.
  --class-order TEXT          Default: soldier,small_aircraft,warship,tank,patrol_boat,armored_vehicle.
  --provenance-manifest PATH  Default: dataset_manifest.csv next to data.yaml if present.
  --seed N                    Default: 42.
  --patience N                Default: 10.
  --image-size N              Default: 640.
  --learning-rate X           Default: 0.001.
  --stop-after-stage N        Default: 6.
  --no-sparse-moe             Disable Sparse-MoE options.
  --builtin-aug               Keep Ultralytics online augmentation enabled.
  --no-amp                    Pass --no-amp to training.
  --no-plots                  Pass --no-plots to training.
  --dry-run                   Pass --dry-run to training.
EOF
}

fail() {
  echo "run_class_il_training.sh: $*" >&2
  exit 2
}

is_nonempty_dir() {
  local path="$1"
  [[ -d "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PRIVATE_ROOT=""
DATA_PATH=""
PREPARED_PATH=""
INITIAL_MODEL=""
OUTPUT_ROOT=""
DER_OUTPUT=""
ER_OUTPUT=""
DATA_NAME="yolo_r1_r2inc_augmented_full_tvt_seed42"
PREPARED_NAME="class_il_prepared_sparse_moe_seed42"
MODEL_NAME="yolo26n.pt"

METHOD="both"
RUN_PREPARE=1
RUN_TRAIN=1
PREPARE_BUFFER_SIZES=()
TRAIN_BUFFER_SIZE=500
CLASS_ORDER="soldier,small_aircraft,warship,tank,patrol_boat,armored_vehicle"
PROVENANCE_MANIFEST=""
SEED=42
EPOCHS=30
PATIENCE=10
IMAGE_SIZE=640
BATCH_SIZE=2
WORKERS=0
DEVICE="cpu"
LEARNING_RATE="0.001"
STOP_AFTER_STAGE=6
SPARSE_MOE=1
EXPERT_COUNT=5
TOP_K=2
EXPERT_BOTTLENECK="0.25"
ROUTER_HIDDEN=128
AUX_HIDDEN=128
MODALITY_LOSS_WEIGHT="0.10"
SCENE_LOSS_WEIGHT="0.10"
BALANCE_LOSS_WEIGHT="0.01"
ROUTER_Z_LOSS_WEIGHT="0.001"
ANCHOR_LOSS_WEIGHT="0.001"
ANCHOR_RHO="0.95"
ROUTER_TEMPERATURE_START="2.0"
ROUTER_TEMPERATURE_END="1.0"
ROUTER_TEMPERATURE_WARMUP_EPOCHS=3
NO_BUILTIN_AUG=1
NO_AMP=0
NO_PLOTS=0
DRY_RUN=0

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
    --private-root)
      PRIVATE_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --data|--data-root|--data-yaml)
      DATA_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --prepared)
      PREPARED_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --initial-model|--model)
      INITIAL_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --der-output)
      DER_OUTPUT="${2:?missing value for $1}"
      shift 2
      ;;
    --er-output)
      ER_OUTPUT="${2:?missing value for $1}"
      shift 2
      ;;
    --data-name)
      DATA_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --prepared-name)
      PREPARED_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --method)
      METHOD="${2:?missing value for $1}"
      shift 2
      ;;
    --skip-prepare|--train-only)
      RUN_PREPARE=0
      RUN_TRAIN=1
      shift
      ;;
    --prepare-only)
      RUN_PREPARE=1
      RUN_TRAIN=0
      shift
      ;;
    --prepare-buffer-size)
      PREPARE_BUFFER_SIZES+=("${2:?missing value for $1}")
      shift 2
      ;;
    --buffer-size)
      TRAIN_BUFFER_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --class-order)
      CLASS_ORDER="${2:?missing value for $1}"
      shift 2
      ;;
    --provenance-manifest)
      PROVENANCE_MANIFEST="${2:?missing value for $1}"
      shift 2
      ;;
    --seed)
      SEED="${2:?missing value for $1}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?missing value for $1}"
      shift 2
      ;;
    --patience)
      PATIENCE="${2:?missing value for $1}"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:?missing value for $1}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?missing value for $1}"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="${2:?missing value for $1}"
      shift 2
      ;;
    --stop-after-stage)
      STOP_AFTER_STAGE="${2:?missing value for $1}"
      shift 2
      ;;
    --smoke-test)
      EPOCHS=1
      STOP_AFTER_STAGE=1
      shift
      ;;
    --no-sparse-moe)
      SPARSE_MOE=0
      shift
      ;;
    --expert-count)
      EXPERT_COUNT="${2:?missing value for $1}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2:?missing value for $1}"
      shift 2
      ;;
    --expert-bottleneck)
      EXPERT_BOTTLENECK="${2:?missing value for $1}"
      shift 2
      ;;
    --router-hidden)
      ROUTER_HIDDEN="${2:?missing value for $1}"
      shift 2
      ;;
    --aux-hidden)
      AUX_HIDDEN="${2:?missing value for $1}"
      shift 2
      ;;
    --modality-loss-weight)
      MODALITY_LOSS_WEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --scene-loss-weight)
      SCENE_LOSS_WEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --balance-loss-weight)
      BALANCE_LOSS_WEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --router-z-loss-weight)
      ROUTER_Z_LOSS_WEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --anchor-loss-weight)
      ANCHOR_LOSS_WEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --anchor-rho)
      ANCHOR_RHO="${2:?missing value for $1}"
      shift 2
      ;;
    --router-temperature-start)
      ROUTER_TEMPERATURE_START="${2:?missing value for $1}"
      shift 2
      ;;
    --router-temperature-end)
      ROUTER_TEMPERATURE_END="${2:?missing value for $1}"
      shift 2
      ;;
    --router-temperature-warmup-epochs)
      ROUTER_TEMPERATURE_WARMUP_EPOCHS="${2:?missing value for $1}"
      shift 2
      ;;
    --builtin-aug)
      NO_BUILTIN_AUG=0
      shift
      ;;
    --no-amp)
      NO_AMP=1
      shift
      ;;
    --no-plots)
      NO_PLOTS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

case "${METHOD}" in
  der|er|both) ;;
  *) fail "--method must be one of: der, er, both" ;;
esac

if [[ "${#PREPARE_BUFFER_SIZES[@]}" -eq 0 ]]; then
  PREPARE_BUFFER_SIZES=(200 500)
fi

if [[ -n "${PRIVATE_ROOT}" ]]; then
  [[ -n "${DATA_PATH}" ]] || DATA_PATH="${PRIVATE_ROOT}/${DATA_NAME}"
  [[ -n "${PREPARED_PATH}" ]] || PREPARED_PATH="${PRIVATE_ROOT}/${PREPARED_NAME}"
  [[ -n "${INITIAL_MODEL}" ]] || INITIAL_MODEL="${PRIVATE_ROOT}/${MODEL_NAME}"
  [[ -n "${OUTPUT_ROOT}" ]] || OUTPUT_ROOT="${PRIVATE_ROOT}/runs"
fi

if [[ "${RUN_PREPARE}" -eq 1 ]]; then
  [[ -n "${DATA_PATH}" ]] || fail "missing --data or --private-root"
fi
[[ -n "${PREPARED_PATH}" ]] || fail "missing --prepared or --private-root"
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  [[ -n "${INITIAL_MODEL}" ]] || fail "missing --initial-model or --private-root"
fi
[[ -n "${OUTPUT_ROOT}" ]] || OUTPUT_ROOT="${PROJECT_ROOT}/outputs/class_il"

DATA_ROOT=""
DATA_YAML=""
if [[ -n "${DATA_PATH}" ]]; then
  if [[ -d "${DATA_PATH}" ]]; then
    DATA_ROOT="${DATA_PATH}"
    DATA_YAML="${DATA_PATH}/data.yaml"
  else
    DATA_YAML="${DATA_PATH}"
    DATA_ROOT="$(dirname "${DATA_YAML}")"
  fi
  [[ -f "${DATA_YAML}" ]] || fail "data.yaml not found: ${DATA_YAML}"
  DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
fi
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  [[ -f "${INITIAL_MODEL}" ]] || fail "initial model not found: ${INITIAL_MODEL}"
fi

if [[ "${RUN_PREPARE}" -eq 1 && -z "${PROVENANCE_MANIFEST}" && -f "${DATA_ROOT}/dataset_manifest.csv" ]]; then
  PROVENANCE_MANIFEST="${DATA_ROOT}/dataset_manifest.csv"
fi
if [[ "${RUN_PREPARE}" -eq 1 && -n "${PROVENANCE_MANIFEST}" ]]; then
  [[ -f "${PROVENANCE_MANIFEST}" ]] || fail "provenance manifest not found: ${PROVENANCE_MANIFEST}"
fi

if [[ "${RUN_PREPARE}" -eq 1 ]] && is_nonempty_dir "${PREPARED_PATH}"; then
  fail "prepared directory is not empty: ${PREPARED_PATH}
Use --skip-prepare to reuse it, or pass a new --prepared path."
fi

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  [[ -n "${DER_OUTPUT}" ]] || DER_OUTPUT="${OUTPUT_ROOT}/class_il_sparse_moe_der_b${TRAIN_BUFFER_SIZE}_e${EPOCHS}_seed${SEED}"
  [[ -n "${ER_OUTPUT}" ]] || ER_OUTPUT="${OUTPUT_ROOT}/class_il_sparse_moe_er_b${TRAIN_BUFFER_SIZE}_e${EPOCHS}_seed${SEED}"
  if [[ "${METHOD}" == "der" || "${METHOD}" == "both" ]] && is_nonempty_dir "${DER_OUTPUT}"; then
    fail "DER output directory is not empty: ${DER_OUTPUT}
Pass a new --der-output or --output-root."
  fi
  if [[ "${METHOD}" == "er" || "${METHOD}" == "both" ]] && is_nonempty_dir "${ER_OUTPUT}"; then
    fail "ER output directory is not empty: ${ER_OUTPUT}
Pass a new --er-output or --output-root."
  fi
fi

if [[ "${DEVICE}" == "0" ]]; then
  echo "[WARN] --device 0 usually means CUDA device 0, not Ascend NPU."
  echo "[WARN] Use --device cpu for the supported board path, or --device npu:0 only if torch_npu training is verified."
fi

if [[ "${DEVICE}" == npu* ]]; then
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  fi
  echo "[WARN] Ascend 310B NPU training depends on a working torch_npu/Ultralytics stack and may be experimental."
fi

cd "${PROJECT_ROOT}"

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Python       : ${PYTHON_BIN}"
if [[ -n "${DATA_YAML}" ]]; then
  echo "[INFO] Data YAML    : ${DATA_YAML}"
fi
echo "[INFO] Prepared     : ${PREPARED_PATH}"
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  echo "[INFO] Initial model: ${INITIAL_MODEL}"
  echo "[INFO] Method       : ${METHOD}"
  echo "[INFO] Device       : ${DEVICE}"
  echo "[INFO] Batch/workers: ${BATCH_SIZE}/${WORKERS}"
fi

if [[ "${RUN_PREPARE}" -eq 1 ]]; then
  prepare_cmd=(
    "${PYTHON_BIN}" train.py prepare-class-il
    --data "${DATA_YAML}"
    --output "${PREPARED_PATH}"
    --seed "${SEED}"
    --class-order "${CLASS_ORDER}"
  )
  for size in "${PREPARE_BUFFER_SIZES[@]}"; do
    prepare_cmd+=(--buffer-size "${size}")
  done
  if [[ -n "${PROVENANCE_MANIFEST}" ]]; then
    prepare_cmd+=(--provenance-manifest "${PROVENANCE_MANIFEST}")
  fi

  echo "[INFO] Preparing Class-IL dataset..."
  "${prepare_cmd[@]}"
fi

run_train() {
  local method="$1"
  local output="$2"
  local cmd=(
    "${PYTHON_BIN}" train.py class-il-yolo
    --prepared "${PREPARED_PATH}"
    --initial-model "${INITIAL_MODEL}"
    --method "${method}"
    --buffer-size "${TRAIN_BUFFER_SIZE}"
    --output "${output}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --image-size "${IMAGE_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
    --stop-after-stage "${STOP_AFTER_STAGE}"
  )

  if [[ "${SPARSE_MOE}" -eq 1 ]]; then
    cmd+=(
      --sparse-moe
      --expert-count "${EXPERT_COUNT}"
      --top-k "${TOP_K}"
      --expert-bottleneck "${EXPERT_BOTTLENECK}"
      --router-hidden "${ROUTER_HIDDEN}"
      --aux-hidden "${AUX_HIDDEN}"
      --modality-loss-weight "${MODALITY_LOSS_WEIGHT}"
      --scene-loss-weight "${SCENE_LOSS_WEIGHT}"
      --balance-loss-weight "${BALANCE_LOSS_WEIGHT}"
      --router-z-loss-weight "${ROUTER_Z_LOSS_WEIGHT}"
      --anchor-loss-weight "${ANCHOR_LOSS_WEIGHT}"
      --anchor-rho "${ANCHOR_RHO}"
      --router-temperature-start "${ROUTER_TEMPERATURE_START}"
      --router-temperature-end "${ROUTER_TEMPERATURE_END}"
      --router-temperature-warmup-epochs "${ROUTER_TEMPERATURE_WARMUP_EPOCHS}"
    )
  fi

  if [[ "${method}" == "der" ]]; then
    cmd+=(
      --der-weight 1.0
      --der-cls-weight 1.0
      --der-box-weight 0.25
      --der-min-confidence 0.0
    )
  fi
  [[ "${NO_BUILTIN_AUG}" -eq 1 ]] && cmd+=(--no-builtin-aug)
  [[ "${NO_AMP}" -eq 1 ]] && cmd+=(--no-amp)
  [[ "${NO_PLOTS}" -eq 1 ]] && cmd+=(--no-plots)
  [[ "${DRY_RUN}" -eq 1 ]] && cmd+=(--dry-run)

  echo "[INFO] Training ${method^^}: ${output}"
  "${cmd[@]}"
  echo "[INFO] ${method^^} best weights: ${output}/stage_06_armored_vehicle/weights/best.pt"
}

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  case "${METHOD}" in
    der)
      run_train der "${DER_OUTPUT}"
      ;;
    er)
      run_train er "${ER_OUTPUT}"
      ;;
    both)
      run_train der "${DER_OUTPUT}"
      run_train er "${ER_OUTPUT}"
      ;;
  esac
fi

echo "[INFO] Done."
