#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deployment/ascend310bpro/run_tzb_submission.sh \
    --base-data /path/to/base_test_data.yaml \
    --increment-data /path/to/increment_test_data.yaml \
    --before-model /path/to/increment_before.pt \
    --after-model /path/to/increment_after.pt \
    --npu-summary outputs/ascend310bpro_full/best_6class_infer/summary.json \
    --agent-summary outputs/ascend310bpro_full/agent_reports/agent_summary.json \
    --output-dir outputs/ascend310bpro_full/tzb_submission \
    --device cpu

Use --skip-map to only package FPS/Agent summaries when the official
evaluate_tzb.py script will compute mAP/KRR/New-mAP separately.
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
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_tzb_submission.py" "$@"
