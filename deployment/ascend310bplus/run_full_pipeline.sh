#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bplus/run_full_pipeline.sh \
    --data /path/to/original_yolo_dataset_or_data.yaml \
    --workspace outputs/ascend310bplus_full \
    --soc-version Ascend310B4

Default flow:
  1. Build an offline augmented YOLO dataset from --data.
  2. Convert ONNX models to cached OM files.
  3. Run NPU inference.

Common options:
  --data PATH                  Original YOLO dataset root or data.yaml.
  --workspace PATH             Pipeline workspace. Default: outputs/ascend310bplus_full.
  --classes PATH               Six-class names file. Default: deployment/ascend310bplus/classes_6.txt.
  --augmented-data PATH        Augmented dataset output. Default: <workspace>/augmented_dataset.
  --infer-input PATH           Image/directory/dataset to infer. Default: augmented dataset.
  --output-dir PATH            Inference output. Default: <workspace>/best_6class_infer,
                               or <workspace>/routed_infer with --routed.
  --soc-version VERSION        Default: SOC_VERSION env, otherwise Ascend310B4.
  --default-modality ir|sar    Used when image filenames do not start with ir_ or sar_.
  --split NAME                 Split name under a structured dataset root. Default: train.
  --val-root PATH              Optional independent YOLO validation root for flat datasets.
  --val-images PATH            Optional independent validation image directory/list/file.
  --val-split NAME             Validation split name under --val-root. Default: val.
  --reuse-augmented            Reuse <augmented-data>/data.yaml if it exists.
  --rebuild-augmented          Move existing augmented output aside and rebuild
                               a fresh dataset at the same path.

Stage controls:
  --augment-only               Stop after augmentation.
  --convert-only               Stop after model conversion.
  --infer-only                 Skip augmentation and conversion; only run inference.
  --skip-augment               Use an existing --augmented-data.
  --skip-convert               Let inference reuse existing OM or auto-convert when needed.
  --skip-infer                 Do not run inference.

Optional board-side training:
  --train                      After augmentation, run Class-IL training on the augmented data.
  --initial-model PATH         Required by --train. Local .pt or .yaml initial YOLO model.
  --prepared PATH              Prepared Class-IL directory. Default: <workspace>/class_il_prepared.
  --training-method der|er|both
                               Default: der.
  --train-device DEVICE        Default: cpu. Use npu:0 only with verified torch_npu training.
  --train-batch-size N         Default: 2.
  --train-workers N            Default: 0.
  --train-epochs N             Default: 30.
  --skip-class-il-prepare      Reuse existing prepared Class-IL directory.

Model/path overrides:
  --single-model PATH, --best-model PATH
                               Use one compact six-class detector such as best.onnx.
                               Default: first existing best.onnx under
                               ascend310bplus/models, deployment, then
                               project-root models.
  --routed                     Use the older scene/easy/hard routed deployment instead.
  --single-width N             Single detector input width. Default: auto-detect from ONNX.
  --single-height N            Single detector input height. Default: auto-detect from ONNX.
  --confidence X, --det-conf X Detection confidence for single mode; routed mode passes --det-conf.
  --iou X                      NMS IoU for single mode; hard-branch IoU for routed mode.
  --single-apply-nms           Apply class-wise NMS to compact single-model outputs.
  --scene-model PATH
  --easy-model PATH
  --hard-model PATH
  --config PATH                Routed inference config. Default: deployment/ascend310bplus/config.json.
  --python PATH                Python executable. Default: python3.
  -h, --help                   Show this help.
EOF
}

fail() {
  echo "run_full_pipeline.sh: $*" >&2
  exit 2
}

is_nonempty_dir() {
  local path="$1"
  [[ -d "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

resolve_initial_model_file() {
  local value="$1"
  local candidate

  case "${value}" in
    /path/to/*|path/to/*)
      return 1
      ;;
  esac

  if [[ "${value}" = /* ]]; then
    [[ -f "${value}" ]] && printf '%s\n' "${value}" && return 0
    return 1
  fi

  for candidate in \
    "${value}" \
    "${PROJECT_ROOT}/${value}" \
    "${SCRIPT_DIR}/${value}" \
    "${SCRIPT_DIR}/models/${value}" \
    "${PROJECT_ROOT}/models/${value}" \
    "${PROJECT_ROOT}/deployment/${value}" \
    "${PROJECT_ROOT}/deployment/ascend310bplus/models/${value}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

default_initial_model_path() {
  local name
  local resolved
  for name in yolo26n.pt yolov8n.pt best.pt; do
    if resolved="$(resolve_initial_model_file "${name}")"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  return 1
}

initial_model_hint() {
  cat <<EOF
Initial training needs a trainable YOLO .pt/.yaml file, not .onnx/.om.
Put a model in one of these common locations, or pass --initial-model with the real path:
  ${PROJECT_ROOT}/yolo26n.pt
  ${PROJECT_ROOT}/models/yolo26n.pt
  ${PROJECT_ROOT}/deployment/yolo26n.pt
  ${SCRIPT_DIR}/models/yolo26n.pt

Examples:
  --initial-model models/yolo26n.pt
  --initial-model /home/HwHiAiUser/Desktop/workspace/ChallengeCup/models/yolo26n.pt
EOF
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH=""
WORKSPACE="${PROJECT_ROOT}/outputs/ascend310bplus_full"
CLASSES_PATH="${SCRIPT_DIR}/classes_6.txt"
AUGMENTED_DATA=""
INFER_INPUT=""
INFER_OUTPUT=""
SOC_VERSION_VALUE="${SOC_VERSION:-Ascend310B4}"
DEFAULT_MODALITY=""
SPLIT_NAME="train"
VAL_ROOT=""
VAL_IMAGES=""
VAL_SPLIT="val"
REUSE_AUGMENTED=0
REBUILD_AUGMENTED=0

RUN_AUGMENT=1
RUN_CONVERT=1
RUN_INFER=1
RUN_TRAIN=0

INITIAL_MODEL=""
PREPARED_PATH=""
TRAINING_METHOD="der"
TRAIN_DEVICE="cpu"
TRAIN_BATCH_SIZE=2
TRAIN_WORKERS=0
TRAIN_EPOCHS=30
SKIP_CLASS_IL_PREPARE=0

CONFIG_PATH="${SCRIPT_DIR}/config.json"
SINGLE_MODEL="$(default_best_model_path)"
SINGLE_WIDTH=""
SINGLE_HEIGHT=""
SINGLE_APPLY_NMS=0
SCENE_MODEL=""
EASY_MODEL=""
HARD_MODEL=""
FORCE_CONVERT=0
NO_SAVE_IMAGES=0
ROUTE_CONFIDENCE=""
DET_CONF=""
EASY_CONF=""
HARD_CONF=""
HARD_IOU=""
HARD_SCORE_ACTIVATION=""

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
    --data|--data-root|--data-yaml)
      DATA_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --workspace)
      WORKSPACE="${2:?missing value for $1}"
      shift 2
      ;;
    --classes)
      CLASSES_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --augmented-data)
      AUGMENTED_DATA="${2:?missing value for $1}"
      shift 2
      ;;
    --infer-input)
      INFER_INPUT="${2:?missing value for $1}"
      shift 2
      ;;
    --output-dir)
      INFER_OUTPUT="${2:?missing value for $1}"
      shift 2
      ;;
    --soc-version)
      SOC_VERSION_VALUE="${2:?missing value for $1}"
      shift 2
      ;;
    --default-modality)
      DEFAULT_MODALITY="${2:?missing value for $1}"
      shift 2
      ;;
    --split)
      SPLIT_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --val-root)
      VAL_ROOT="${2:?missing value for $1}"
      shift 2
      ;;
    --val-images)
      VAL_IMAGES="${2:?missing value for $1}"
      shift 2
      ;;
    --val-split)
      VAL_SPLIT="${2:?missing value for $1}"
      shift 2
      ;;
    --reuse-augmented)
      REUSE_AUGMENTED=1
      shift
      ;;
    --rebuild-augmented)
      REBUILD_AUGMENTED=1
      shift
      ;;
    --augment-only)
      RUN_AUGMENT=1
      RUN_CONVERT=0
      RUN_INFER=0
      RUN_TRAIN=0
      shift
      ;;
    --convert-only)
      RUN_AUGMENT=0
      RUN_CONVERT=1
      RUN_INFER=0
      RUN_TRAIN=0
      shift
      ;;
    --infer-only)
      RUN_AUGMENT=0
      RUN_CONVERT=0
      RUN_INFER=1
      RUN_TRAIN=0
      shift
      ;;
    --skip-augment)
      RUN_AUGMENT=0
      shift
      ;;
    --skip-convert)
      RUN_CONVERT=0
      shift
      ;;
    --skip-infer)
      RUN_INFER=0
      shift
      ;;
    --train)
      RUN_TRAIN=1
      shift
      ;;
    --initial-model|--model)
      INITIAL_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --prepared)
      PREPARED_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --training-method)
      TRAINING_METHOD="${2:?missing value for $1}"
      shift 2
      ;;
    --train-device)
      TRAIN_DEVICE="${2:?missing value for $1}"
      shift 2
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --train-workers)
      TRAIN_WORKERS="${2:?missing value for $1}"
      shift 2
      ;;
    --train-epochs)
      TRAIN_EPOCHS="${2:?missing value for $1}"
      shift 2
      ;;
    --skip-class-il-prepare)
      SKIP_CLASS_IL_PREPARE=1
      shift
      ;;
    --config)
      CONFIG_PATH="${2:?missing value for $1}"
      shift 2
      ;;
    --single-model|--best-model)
      SINGLE_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --routed)
      SINGLE_MODEL=""
      shift
      ;;
    --single-width)
      SINGLE_WIDTH="${2:?missing value for $1}"
      shift 2
      ;;
    --single-height)
      SINGLE_HEIGHT="${2:?missing value for $1}"
      shift 2
      ;;
    --confidence|--det-conf)
      DET_CONF="${2:?missing value for $1}"
      shift 2
      ;;
    --single-apply-nms)
      SINGLE_APPLY_NMS=1
      shift
      ;;
    --scene-model)
      SCENE_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --easy-model)
      EASY_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --hard-model)
      HARD_MODEL="${2:?missing value for $1}"
      shift 2
      ;;
    --force-convert)
      FORCE_CONVERT=1
      shift
      ;;
    --no-save-images)
      NO_SAVE_IMAGES=1
      shift
      ;;
    --route-confidence)
      ROUTE_CONFIDENCE="${2:?missing value for $1}"
      shift 2
      ;;
    --easy-conf)
      EASY_CONF="${2:?missing value for $1}"
      shift 2
      ;;
    --hard-conf)
      HARD_CONF="${2:?missing value for $1}"
      shift 2
      ;;
    --hard-iou|--iou)
      HARD_IOU="${2:?missing value for $1}"
      shift 2
      ;;
    --hard-score-activation)
      HARD_SCORE_ACTIVATION="${2:?missing value for $1}"
      shift 2
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

case "${DEFAULT_MODALITY}" in
  ""|ir|sar) ;;
  *) fail "--default-modality must be ir or sar" ;;
esac
if [[ -n "${VAL_ROOT}" && -n "${VAL_IMAGES}" ]]; then
  fail "use either --val-root or --val-images, not both"
fi
if [[ -n "${SINGLE_WIDTH}" && -z "${SINGLE_HEIGHT}" ]] || [[ -z "${SINGLE_WIDTH}" && -n "${SINGLE_HEIGHT}" ]]; then
  fail "pass both --single-width and --single-height, or neither"
fi
if [[ "${REUSE_AUGMENTED}" -eq 1 && "${REBUILD_AUGMENTED}" -eq 1 ]]; then
  fail "use either --reuse-augmented or --rebuild-augmented, not both"
fi
case "${TRAINING_METHOD}" in
  der|er|both) ;;
  *) fail "--training-method must be one of: der, er, both" ;;
esac

if [[ "${RUN_TRAIN}" -eq 1 && -z "${INITIAL_MODEL}" ]]; then
  if INITIAL_MODEL="$(default_initial_model_path)"; then
    echo "[INFO] Auto-detected initial model: ${INITIAL_MODEL}"
  else
    fail "--train requires --initial-model
$(initial_model_hint)"
  fi
fi
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  case "${INITIAL_MODEL}" in
    *.onnx|*.om)
      fail "training cannot start from deployment model: ${INITIAL_MODEL}
$(initial_model_hint)"
      ;;
  esac
  if resolved_initial_model="$(resolve_initial_model_file "${INITIAL_MODEL}")"; then
    INITIAL_MODEL="${resolved_initial_model}"
  else
    fail "initial model not found: ${INITIAL_MODEL}
$(initial_model_hint)"
  fi
fi

[[ -n "${AUGMENTED_DATA}" ]] || AUGMENTED_DATA="${WORKSPACE}/augmented_dataset"
if [[ -z "${INFER_OUTPUT}" ]]; then
  if [[ -n "${SINGLE_MODEL}" ]]; then
    INFER_OUTPUT="${WORKSPACE}/best_6class_infer"
  else
    INFER_OUTPUT="${WORKSPACE}/routed_infer"
  fi
fi
[[ -n "${PREPARED_PATH}" ]] || PREPARED_PATH="${WORKSPACE}/class_il_prepared"

if [[ "${RUN_AUGMENT}" -eq 1 ]]; then
  [[ -n "${DATA_PATH}" ]] || fail "missing --data for augmentation"
fi
if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  [[ -n "${INITIAL_MODEL}" ]] || fail "--train requires --initial-model"
fi
if [[ "${RUN_INFER}" -eq 1 && -z "${INFER_INPUT}" ]]; then
  INFER_INPUT="${AUGMENTED_DATA}"
fi
if [[ "${RUN_INFER}" -eq 1 && "${RUN_AUGMENT}" -eq 0 && ! -e "${INFER_INPUT}" ]]; then
  fail "inference input not found: ${INFER_INPUT}"
fi
if [[ "${RUN_TRAIN}" -eq 1 && "${RUN_AUGMENT}" -eq 0 && ! -f "${AUGMENTED_DATA}/data.yaml" ]]; then
  fail "training needs augmented data.yaml: ${AUGMENTED_DATA}/data.yaml"
fi

mkdir -p "${WORKSPACE}"
export PYTHON_BIN
[[ -f "${CLASSES_PATH}" ]] || fail "classes file not found: ${CLASSES_PATH}"

echo "[INFO] Project root : ${PROJECT_ROOT}"
echo "[INFO] Workspace    : ${WORKSPACE}"
echo "[INFO] Classes      : ${CLASSES_PATH}"

if [[ "${RUN_AUGMENT}" -eq 1 ]]; then
  augment_cmd=(
    bash "${SCRIPT_DIR}/run_augment_yolo.sh"
    --python "${PYTHON_BIN}"
    --data "${DATA_PATH}"
    --output "${AUGMENTED_DATA}"
    --classes "${CLASSES_PATH}"
    --split "${SPLIT_NAME}"
  )
  if [[ -n "${DEFAULT_MODALITY}" ]]; then
    augment_cmd+=(--default-modality "${DEFAULT_MODALITY}")
  fi
  if [[ -n "${VAL_ROOT}" ]]; then
    augment_cmd+=(--val-root "${VAL_ROOT}" --val-split "${VAL_SPLIT}")
  fi
  if [[ -n "${VAL_IMAGES}" ]]; then
    augment_cmd+=(--val-images "${VAL_IMAGES}")
  fi
  if [[ "${REUSE_AUGMENTED}" -eq 1 ]]; then
    augment_cmd+=(--reuse-existing)
  fi
  if [[ "${REBUILD_AUGMENTED}" -eq 1 ]]; then
    augment_cmd+=(--rebuild-existing)
  fi
  echo "[INFO] Stage 1/4: dataset augmentation"
  "${augment_cmd[@]}"
elif [[ -f "${AUGMENTED_DATA}/data.yaml" ]]; then
  echo "[INFO] Stage 1/4 skipped: reuse augmented dataset ${AUGMENTED_DATA}/data.yaml"
else
  echo "[INFO] Stage 1/4 skipped: no augmented dataset check requested"
fi

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  [[ -f "${AUGMENTED_DATA}/data.yaml" ]] || fail "training needs augmented data.yaml: ${AUGMENTED_DATA}/data.yaml"
  training_cmd=(
    bash "${SCRIPT_DIR}/run_class_il_training.sh"
    --python "${PYTHON_BIN}"
    --data "${AUGMENTED_DATA}/data.yaml"
    --prepared "${PREPARED_PATH}"
    --initial-model "${INITIAL_MODEL}"
    --output-root "${WORKSPACE}/runs"
    --method "${TRAINING_METHOD}"
    --device "${TRAIN_DEVICE}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --workers "${TRAIN_WORKERS}"
    --epochs "${TRAIN_EPOCHS}"
  )
  if [[ "${SKIP_CLASS_IL_PREPARE}" -eq 1 ]]; then
    training_cmd+=(--skip-prepare)
  fi
  echo "[INFO] Stage 2/4: optional Class-IL training"
  "${training_cmd[@]}"
else
  echo "[INFO] Stage 2/4 skipped: board-side training is optional"
fi

model_args=()
if [[ -n "${SCENE_MODEL}" ]]; then
  model_args+=(--scene-model "${SCENE_MODEL}")
fi
if [[ -n "${EASY_MODEL}" ]]; then
  model_args+=(--easy-model "${EASY_MODEL}")
fi
if [[ -n "${HARD_MODEL}" ]]; then
  model_args+=(--hard-model "${HARD_MODEL}")
fi

if [[ "${RUN_CONVERT}" -eq 1 ]]; then
  if [[ -n "${SINGLE_MODEL}" ]]; then
    convert_cmd=(
      bash "${SCRIPT_DIR}/run_best_6class_npu.sh"
      --model "${SINGLE_MODEL}"
      --classes "${CLASSES_PATH}"
      --soc-version "${SOC_VERSION_VALUE}"
      --convert-only
    )
    if [[ -n "${SINGLE_WIDTH}" ]]; then
      convert_cmd+=(--width "${SINGLE_WIDTH}" --height "${SINGLE_HEIGHT}")
    fi
  else
    convert_cmd=(
      bash "${SCRIPT_DIR}/convert_models.sh"
      --config "${CONFIG_PATH}"
      --soc-version "${SOC_VERSION_VALUE}"
      "${model_args[@]}"
    )
  fi
  if [[ "${FORCE_CONVERT}" -eq 1 ]]; then
    convert_cmd+=(--force-convert)
  fi
  echo "[INFO] Stage 3/4: ONNX to OM conversion"
  "${convert_cmd[@]}"
else
  echo "[INFO] Stage 3/4 skipped: conversion disabled"
fi

if [[ "${RUN_INFER}" -eq 1 ]]; then
  if [[ -n "${SINGLE_MODEL}" ]]; then
    infer_cmd=(
      bash "${SCRIPT_DIR}/run_best_6class_npu.sh"
      --model "${SINGLE_MODEL}"
      --classes "${CLASSES_PATH}"
      --input "${INFER_INPUT}"
      --output-dir "${INFER_OUTPUT}"
      --soc-version "${SOC_VERSION_VALUE}"
    )
    if [[ -n "${SINGLE_WIDTH}" ]]; then
      infer_cmd+=(--width "${SINGLE_WIDTH}" --height "${SINGLE_HEIGHT}")
    fi
  else
    infer_cmd=(
      bash "${SCRIPT_DIR}/run_routed_infer.sh"
      --config "${CONFIG_PATH}"
      --input "${INFER_INPUT}"
      --output-dir "${INFER_OUTPUT}"
      --soc-version "${SOC_VERSION_VALUE}"
      "${model_args[@]}"
    )
  fi
  if [[ "${NO_SAVE_IMAGES}" -eq 1 ]]; then
    infer_cmd+=(--no-save-images)
  fi
  if [[ -n "${DET_CONF}" ]]; then
    if [[ -n "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--confidence "${DET_CONF}")
    else
      infer_cmd+=(--det-conf "${DET_CONF}")
    fi
  fi
  if [[ -n "${ROUTE_CONFIDENCE}" ]]; then
    if [[ -z "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--route-confidence "${ROUTE_CONFIDENCE}")
    fi
  fi
  if [[ -n "${EASY_CONF}" ]]; then
    if [[ -z "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--easy-conf "${EASY_CONF}")
    fi
  fi
  if [[ -n "${HARD_CONF}" ]]; then
    if [[ -z "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--hard-conf "${HARD_CONF}")
    fi
  fi
  if [[ -n "${HARD_IOU}" ]]; then
    if [[ -n "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--iou "${HARD_IOU}")
    else
      infer_cmd+=(--hard-iou "${HARD_IOU}")
    fi
  fi
  if [[ -n "${HARD_SCORE_ACTIVATION}" ]]; then
    if [[ -z "${SINGLE_MODEL}" ]]; then
      infer_cmd+=(--hard-score-activation "${HARD_SCORE_ACTIVATION}")
    fi
  fi
  if [[ "${SINGLE_APPLY_NMS}" -eq 1 && -n "${SINGLE_MODEL}" ]]; then
    infer_cmd+=(--apply-nms)
  fi
  if [[ -n "${SINGLE_MODEL}" ]]; then
    echo "[INFO] Stage 4/4: single six-class best.onnx inference"
  else
    echo "[INFO] Stage 4/4: routed inference, scene recognition -> model matching"
  fi
  "${infer_cmd[@]}"
  echo "[INFO] Inference summary: ${INFER_OUTPUT}/summary.json"
  echo "[INFO] Inference JSONL  : ${INFER_OUTPUT}/predictions.jsonl"
else
  echo "[INFO] Stage 4/4 skipped: inference disabled"
fi

if [[ "${RUN_AUGMENT}" -eq 1 || -f "${AUGMENTED_DATA}/data.yaml" ]]; then
  echo "[INFO] Augmented data.yaml: ${AUGMENTED_DATA}/data.yaml"
fi
echo "[INFO] Done."
