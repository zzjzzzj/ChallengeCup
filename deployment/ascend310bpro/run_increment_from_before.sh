#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_increment_from_before.sh \
    --base-data outputs/ascend310bpro_before_pc/base_before_augmented/data.yaml \
    --increment-data /path/to/six_class_increment_train/data.yaml \
    --before-model deployment/ascend310bpro/models/before_increment_4class.pt \
    --workspace outputs/ascend310bpro_increment \
    --method der \
    --device cpu

Purpose:
  Continue from a four-class increment-before .pt checkpoint and train the
  six-class increment-after model on the board.

Important:
  --before-model must be a trainable .pt checkpoint. ONNX/OM are inference
  artifacts and cannot be used for incremental training.
  --increment-data must contain labels and a six-class data.yaml:
    soldier, small_aircraft, warship, tank, patrol_boat, armored_vehicle

Options:
  --base-data PATH             Four-class base data.yaml or flat dataset root.
  --increment-data PATH        Six-class increment data.yaml or dataset root.
  --before-model PATH          Four-class before .pt checkpoint.
  --workspace PATH             Workspace. Default: outputs/ascend310bpro_increment.
  --prepared PATH              Prepared protocol dir. Default: WORKSPACE/prepared_batch_il.
  --output PATH                Training output dir. Default: WORKSPACE/runs/batch_il_METHOD_bBUFFER.
  --base-classes PATH          Default: deployment/ascend310bpro/classes_base4.txt.
  --increment-classes PATH     Default: deployment/ascend310bpro/classes_6.txt.
  --method der|er              Default: der.
  --num-batches N              Default: 1.
  --batch-plan PATH            Optional JSON batch plan.
  --prepare-buffer-size N      Repeatable. Default: same as --buffer-size.
  --buffer-size N              Training replay buffer. Default: 500.
  --max-current-images-per-class N
  --skip-prepare               Reuse existing --prepared.
  --prepare-only               Only prepare data, do not train.
  --reuse-augmented            Reuse WORKSPACE/base_augmented or increment_augmented.
  --rebuild-augmented          Rebuild generated augmentation dirs.
  --device DEVICE              Default: cpu. Use npu:0 only with working torch_npu.
  --batch-size N               Default: 2.
  --workers N                  Default: 0.
  --epochs N                   Default: 30.
  --smoke-test                 1 epoch, stop after first batch.
  --no-sparse-moe              Disable Sparse-MoE trainer.
  --no-export-onnx             Do not export final .pt to ONNX.
  --export-image-size N        Default: same as --image-size.
  --export-device DEVICE       Default: cpu.
  --dry-run                    Validate plan without training.
EOF
}

fail() {
  echo "run_increment_from_before.sh: $*" >&2
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

direct_data_yaml() {
  local value="$1"
  local path
  path="$(abspath "${value}")"
  if [[ -f "${path}" ]]; then
    printf '%s\n' "${path}"
    return 0
  fi
  if [[ -f "${path}/data.yaml" ]]; then
    printf '%s\n' "${path}/data.yaml"
    return 0
  fi
  return 1
}

augment_if_needed() {
  local input_path="$1"
  local output_path="$2"
  local classes_path="$3"
  local label="$4"

  if [[ -f "${output_path}/data.yaml" && "${REUSE_AUGMENTED}" -eq 1 && "${REBUILD_AUGMENTED}" -eq 0 ]]; then
    echo "[INFO] Reuse ${label} augmented dataset: ${output_path}/data.yaml" >&2
    printf '%s\n' "${output_path}/data.yaml"
    return 0
  fi

  local cmd=(bash "${SCRIPT_DIR}/run_augment_yolo.sh" --data "${input_path}" --output "${output_path}" --classes "${classes_path}")
  if [[ "${REBUILD_AUGMENTED}" -eq 1 ]]; then
    cmd+=(--rebuild-existing)
  else
    cmd+=(--reuse-existing)
  fi
  "${cmd[@]}" >&2
  printf '%s\n' "${output_path}/data.yaml"
}

BASE_DATA=""
INCREMENT_DATA=""
BEFORE_MODEL=""
WORKSPACE="outputs/ascend310bpro_increment"
PREPARED=""
OUTPUT=""
BASE_CLASSES="${SCRIPT_DIR}/classes_base4.txt"
INCREMENT_CLASSES="${SCRIPT_DIR}/classes_6.txt"
METHOD="der"
NUM_BATCHES=1
BATCH_PLAN=""
PREPARE_BUFFER_SIZES=()
BUFFER_SIZE=500
MAX_CURRENT_IMAGES_PER_CLASS=""
RUN_PREPARE=1
RUN_TRAIN=1
REUSE_AUGMENTED=0
REBUILD_AUGMENTED=0
DEVICE="cpu"
BATCH_SIZE=2
WORKERS=0
EPOCHS=30
PATIENCE=10
IMAGE_SIZE=640
SEED=42
LEARNING_RATE="0.001"
FREEZE=""
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
STOP_AFTER_BATCH=""
EXPORT_ONNX=1
EXPORT_IMAGE_SIZE=""
EXPORT_OPSET=12
EXPORT_DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --python) PYTHON_BIN="${2:?missing value for $1}"; shift 2 ;;
    --base-data) BASE_DATA="${2:?missing value for $1}"; shift 2 ;;
    --increment-data) INCREMENT_DATA="${2:?missing value for $1}"; shift 2 ;;
    --before-model|--initial-checkpoint|--initial-model) BEFORE_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --workspace) WORKSPACE="${2:?missing value for $1}"; shift 2 ;;
    --prepared) PREPARED="${2:?missing value for $1}"; shift 2 ;;
    --output) OUTPUT="${2:?missing value for $1}"; shift 2 ;;
    --base-classes) BASE_CLASSES="${2:?missing value for $1}"; shift 2 ;;
    --increment-classes) INCREMENT_CLASSES="${2:?missing value for $1}"; shift 2 ;;
    --method) METHOD="${2:?missing value for $1}"; shift 2 ;;
    --num-batches) NUM_BATCHES="${2:?missing value for $1}"; shift 2 ;;
    --batch-plan) BATCH_PLAN="${2:?missing value for $1}"; shift 2 ;;
    --prepare-buffer-size) PREPARE_BUFFER_SIZES+=("${2:?missing value for $1}"); shift 2 ;;
    --buffer-size) BUFFER_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --max-current-images-per-class) MAX_CURRENT_IMAGES_PER_CLASS="${2:?missing value for $1}"; shift 2 ;;
    --skip-prepare|--train-only) RUN_PREPARE=0; shift ;;
    --prepare-only) RUN_TRAIN=0; shift ;;
    --reuse-augmented) REUSE_AUGMENTED=1; shift ;;
    --rebuild-augmented) REBUILD_AUGMENTED=1; shift ;;
    --device) DEVICE="${2:?missing value for $1}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --workers) WORKERS="${2:?missing value for $1}"; shift 2 ;;
    --epochs) EPOCHS="${2:?missing value for $1}"; shift 2 ;;
    --patience) PATIENCE="${2:?missing value for $1}"; shift 2 ;;
    --image-size) IMAGE_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --seed) SEED="${2:?missing value for $1}"; shift 2 ;;
    --learning-rate) LEARNING_RATE="${2:?missing value for $1}"; shift 2 ;;
    --freeze) FREEZE="${2:?missing value for $1}"; shift 2 ;;
    --no-sparse-moe) SPARSE_MOE=0; shift ;;
    --expert-count) EXPERT_COUNT="${2:?missing value for $1}"; shift 2 ;;
    --top-k) TOP_K="${2:?missing value for $1}"; shift 2 ;;
    --expert-bottleneck) EXPERT_BOTTLENECK="${2:?missing value for $1}"; shift 2 ;;
    --router-hidden) ROUTER_HIDDEN="${2:?missing value for $1}"; shift 2 ;;
    --aux-hidden) AUX_HIDDEN="${2:?missing value for $1}"; shift 2 ;;
    --modality-loss-weight) MODALITY_LOSS_WEIGHT="${2:?missing value for $1}"; shift 2 ;;
    --scene-loss-weight) SCENE_LOSS_WEIGHT="${2:?missing value for $1}"; shift 2 ;;
    --balance-loss-weight) BALANCE_LOSS_WEIGHT="${2:?missing value for $1}"; shift 2 ;;
    --router-z-loss-weight) ROUTER_Z_LOSS_WEIGHT="${2:?missing value for $1}"; shift 2 ;;
    --anchor-loss-weight) ANCHOR_LOSS_WEIGHT="${2:?missing value for $1}"; shift 2 ;;
    --anchor-rho) ANCHOR_RHO="${2:?missing value for $1}"; shift 2 ;;
    --router-temperature-start) ROUTER_TEMPERATURE_START="${2:?missing value for $1}"; shift 2 ;;
    --router-temperature-end) ROUTER_TEMPERATURE_END="${2:?missing value for $1}"; shift 2 ;;
    --router-temperature-warmup-epochs) ROUTER_TEMPERATURE_WARMUP_EPOCHS="${2:?missing value for $1}"; shift 2 ;;
    --builtin-aug) NO_BUILTIN_AUG=0; shift ;;
    --no-amp) NO_AMP=1; shift ;;
    --no-plots) NO_PLOTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke-test) EPOCHS=1; PATIENCE=1; STOP_AFTER_BATCH=1; shift ;;
    --stop-after-batch) STOP_AFTER_BATCH="${2:?missing value for $1}"; shift 2 ;;
    --no-export-onnx) EXPORT_ONNX=0; shift ;;
    --export-image-size) EXPORT_IMAGE_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --export-opset) EXPORT_OPSET="${2:?missing value for $1}"; shift 2 ;;
    --export-device) EXPORT_DEVICE="${2:?missing value for $1}"; shift 2 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ "${METHOD}" == "der" || "${METHOD}" == "er" ]] || fail "--method must be der or er"
[[ -n "${INCREMENT_DATA}" ]] || fail "missing --increment-data; data/testdata is unlabeled and cannot be used for training"
[[ -n "${BEFORE_MODEL}" ]] || fail "missing --before-model"
[[ "${REUSE_AUGMENTED}" -eq 0 || "${REBUILD_AUGMENTED}" -eq 0 ]] || fail "use either --reuse-augmented or --rebuild-augmented"

WORKSPACE="$(abspath "${WORKSPACE}")"
[[ -n "${PREPARED}" ]] || PREPARED="${WORKSPACE}/prepared_batch_il"
[[ -n "${OUTPUT}" ]] || OUTPUT="${WORKSPACE}/runs/batch_il_${METHOD}_b${BUFFER_SIZE}"
[[ -n "${BASE_DATA}" ]] || BASE_DATA="data/datasets_r1_base_train"
[[ -n "${EXPORT_IMAGE_SIZE}" ]] || EXPORT_IMAGE_SIZE="${IMAGE_SIZE}"

BASE_CLASSES="$(abspath "${BASE_CLASSES}")"
INCREMENT_CLASSES="$(abspath "${INCREMENT_CLASSES}")"
PREPARED="$(abspath "${PREPARED}")"
OUTPUT="$(abspath "${OUTPUT}")"
BEFORE_MODEL="$(abspath "${BEFORE_MODEL}")"

[[ -f "${BASE_CLASSES}" ]] || fail "base classes file not found: ${BASE_CLASSES}"
[[ -f "${INCREMENT_CLASSES}" ]] || fail "increment classes file not found: ${INCREMENT_CLASSES}"
[[ -f "${BEFORE_MODEL}" ]] || fail "before model not found: ${BEFORE_MODEL}"
case "${BEFORE_MODEL}" in
  *.pt) ;;
  *.onnx|*.om) fail "--before-model must be .pt for training; ONNX/OM are for inference only" ;;
  *) fail "--before-model must be a .pt checkpoint: ${BEFORE_MODEL}" ;;
esac

if [[ "${DEVICE}" == npu* ]]; then
  if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  fi
  echo "[WARN] NPU-side PyTorch training requires a working torch_npu/Ultralytics stack."
fi

cd "${PROJECT_ROOT}"
mkdir -p "${WORKSPACE}"

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Python       : ${PYTHON_BIN}"
echo "[INFO] Workspace    : ${WORKSPACE}"
echo "[INFO] Before .pt   : ${BEFORE_MODEL}"
echo "[INFO] Method       : ${METHOD}"
echo "[INFO] Device       : ${DEVICE}"

BASE_DATA_YAML=""
if BASE_DATA_YAML="$(direct_data_yaml "${BASE_DATA}")"; then
  echo "[INFO] Base data.yaml: ${BASE_DATA_YAML}"
else
  BASE_DATA_YAML="$(augment_if_needed "${BASE_DATA}" "${WORKSPACE}/base_augmented" "${BASE_CLASSES}" "base")"
fi

INCREMENT_DATA_YAML=""
if INCREMENT_DATA_YAML="$(direct_data_yaml "${INCREMENT_DATA}")"; then
  echo "[INFO] Increment data.yaml: ${INCREMENT_DATA_YAML}"
else
  INCREMENT_DATA_YAML="$(augment_if_needed "${INCREMENT_DATA}" "${WORKSPACE}/increment_augmented" "${INCREMENT_CLASSES}" "increment")"
fi

if [[ "${RUN_PREPARE}" -eq 1 ]]; then
  prepare_cmd=(
    "${PYTHON_BIN}" train.py prepare-batch-il
    --base-data "${BASE_DATA_YAML}"
    --increment-data "${INCREMENT_DATA_YAML}"
    --output "${PREPARED}"
    --seed "${SEED}"
  )
  if [[ -n "${BATCH_PLAN}" ]]; then
    prepare_cmd+=(--batch-plan "$(abspath "${BATCH_PLAN}")")
  else
    prepare_cmd+=(--num-batches "${NUM_BATCHES}")
  fi
  if [[ "${#PREPARE_BUFFER_SIZES[@]}" -eq 0 ]]; then
    PREPARE_BUFFER_SIZES=("${BUFFER_SIZE}")
  fi
  for size in "${PREPARE_BUFFER_SIZES[@]}"; do
    prepare_cmd+=(--buffer-size "${size}")
  done
  if [[ -n "${MAX_CURRENT_IMAGES_PER_CLASS}" ]]; then
    prepare_cmd+=(--max-current-images-per-class "${MAX_CURRENT_IMAGES_PER_CLASS}")
  fi

  echo "[INFO] Preparing four-to-six incremental protocol..."
  "${prepare_cmd[@]}"
fi

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  train_cmd=(
    "${PYTHON_BIN}" train.py batch-il-yolo
    --prepared "${PREPARED}"
    --initial-checkpoint "${BEFORE_MODEL}"
    --method "${METHOD}"
    --buffer-size "${BUFFER_SIZE}"
    --output "${OUTPUT}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --image-size "${IMAGE_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --learning-rate "${LEARNING_RATE}"
  )
  if [[ -n "${FREEZE}" ]]; then
    train_cmd+=(--freeze "${FREEZE}")
  fi
  if [[ "${METHOD}" == "der" ]]; then
    train_cmd+=(--der-weight 1.0 --der-cls-weight 1.0 --der-box-weight 0.25 --der-min-confidence 0.0)
  fi
  if [[ "${SPARSE_MOE}" -eq 1 ]]; then
    train_cmd+=(
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
  [[ "${NO_BUILTIN_AUG}" -eq 1 ]] && train_cmd+=(--no-builtin-aug)
  [[ "${NO_AMP}" -eq 1 ]] && train_cmd+=(--no-amp)
  [[ "${NO_PLOTS}" -eq 1 ]] && train_cmd+=(--no-plots)
  [[ "${DRY_RUN}" -eq 1 ]] && train_cmd+=(--dry-run)
  [[ -n "${STOP_AFTER_BATCH}" ]] && train_cmd+=(--stop-after-batch "${STOP_AFTER_BATCH}")

  echo "[INFO] Training increment-after model..."
  "${train_cmd[@]}"
  echo "[INFO] Training summary: ${OUTPUT}/batch_incremental_training_summary.json"

  if [[ "${DRY_RUN}" -eq 0 && "${EXPORT_ONNX}" -eq 1 ]]; then
    FINAL_MODEL="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig'))['final_model'])" "${OUTPUT}/batch_incremental_training_summary.json")"
    EXPORT_DIR="${OUTPUT}/exports"
    mkdir -p "${EXPORT_DIR}"
    echo "[INFO] Exporting final .pt to ONNX: ${FINAL_MODEL}"
    "${PYTHON_BIN}" scene_recognition/detector_module/export_detector.py \
      --model "${FINAL_MODEL}" \
      --data "${INCREMENT_DATA_YAML}" \
      --output "${EXPORT_DIR}" \
      --image-size "${EXPORT_IMAGE_SIZE}" \
      --opset "${EXPORT_OPSET}" \
      --device "${EXPORT_DEVICE}" \
      --skip-validation
    if [[ -f "${EXPORT_DIR}/detector_yolov8n_bs1.onnx" ]]; then
      cp -f "${EXPORT_DIR}/detector_yolov8n_bs1.onnx" "${EXPORT_DIR}/after_increment_6class_${EXPORT_IMAGE_SIZE}x${EXPORT_IMAGE_SIZE}.onnx"
      echo "[INFO] After-model ONNX: ${EXPORT_DIR}/after_increment_6class_${EXPORT_IMAGE_SIZE}x${EXPORT_IMAGE_SIZE}.onnx"
    fi
  fi
fi

echo "[INFO] Done."
