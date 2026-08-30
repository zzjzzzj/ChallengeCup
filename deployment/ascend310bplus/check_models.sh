#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python3 "${SCRIPT_DIR}/routed_infer_npu.py" --config "${SCRIPT_DIR}/config.json" --check-models "$@"
