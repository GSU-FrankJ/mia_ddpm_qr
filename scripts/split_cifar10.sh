#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/configs/data_cifar10.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/data/splits"

echo "[split_cifar10] Using config: ${CONFIG_PATH}"
echo "[split_cifar10] Writing splits to: ${OUTPUT_DIR}"

python "${PROJECT_ROOT}/scripts/split_cifar10.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" "$@"


