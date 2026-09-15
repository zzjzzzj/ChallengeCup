#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMAND="${1:-verify}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${COMMAND}" in
  build)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/model_manifest.py" build \
      --config "${SCRIPT_DIR}/route_config.yaml" \
      --output "${SCRIPT_DIR}/model_manifest.json" \
      "$@"
    ;;
  verify)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/model_manifest.py" verify \
      --config "${SCRIPT_DIR}/route_config.yaml" \
      --manifest "${SCRIPT_DIR}/model_manifest.json" \
      "$@"
    ;;
  -h|--help)
    cat <<'EOF'
Usage:
  run_model_manifest.sh build [--soc-version Ascend310B4]
  run_model_manifest.sh verify

The manifest records the routed deployment model hashes, input sizes, class
lists, preprocessing policy, and local-to-global class remap.
EOF
    ;;
  *)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/model_manifest.py" "${COMMAND}" "$@"
    ;;
esac
