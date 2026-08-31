#!/usr/bin/env bash
set -euo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
python3 "${SCRIPT_DIR}/run_end_to_end.py" "$@"
