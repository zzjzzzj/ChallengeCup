#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_agent_reports.sh \
    --predictions outputs/ascend310bpro_full/routed_infer/predictions.jsonl \
    --summary outputs/ascend310bpro_full/routed_infer/summary.json \
    --output-dir outputs/ascend310bpro_full/agent_reports \
    --memory outputs/ascend310bpro_full/agent_memory.jsonl

This does not run NPU inference again. It formats existing predictions into:
  <output-dir>/reports/*.json
  <output-dir>/batch_summary.csv
  <output-dir>/agent_summary.json
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
"${PYTHON_BIN}" "${SCRIPT_DIR}/agent_outputs.py" format "$@"
