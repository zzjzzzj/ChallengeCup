#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_feedback.sh \
    --memory outputs/ascend310bpro_full/agent_memory.jsonl \
    --image data/datasets_r1_base_train/ir_r1_base_air_000001.png \
    --scene air \
    --modality ir \
    --targets small_aircraft \
    --note "manual correction"

The feedback record is appended to the JSONL memory file and can be used later
for sample review or incremental-training replay selection.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/agent_outputs.py" feedback "$@"
