#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"

CONFIG_PATH="${PROJECT_ROOT}/configs/model_ddim.yaml"
DATA_CONFIG="${PROJECT_ROOT}/configs/data_cifar10.yaml"

echo "[train_ddim] Config: ${CONFIG_PATH}"
echo "[train_ddim] Data config: ${DATA_CONFIG}"

python "${PROJECT_ROOT}/ddpm_ddim/train_ddim.py" \
  --config "${CONFIG_PATH}" \
  --data "${DATA_CONFIG}" "$@"

