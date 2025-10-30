#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"

ATTACK_CONFIG="${PROJECT_ROOT}/configs/attack_qr.yaml"
DATA_CONFIG="${PROJECT_ROOT}/configs/data_cifar10.yaml"

echo "[eval_qr] Attack config: ${ATTACK_CONFIG}"

python "${PROJECT_ROOT}/attacks/eval/evaluate.py" \
  --config "${ATTACK_CONFIG}" \
  --data-config "${DATA_CONFIG}" "$@"

