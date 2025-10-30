#!/bin/bash
# Split CIFAR-10 dataset into reproducible splits with no leakage
# Creates aux.json, eval_in.json, eval_out.json in data/splits/

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=== Creating CIFAR-10 splits ==="
echo "Output directory: data/splits/"

# Run the Python script to create splits
python scripts/split_cifar10.py \
    --data-root data/cifar10 \
    --output-dir data/splits \
    --seed 20251030 \
    --member-train-size 40000 \
    --eval-in-size 5000 \
    --eval-out-size 5000 \
    --aux-size 10000

echo ""
echo "=== Split creation complete ==="
echo "Files created:"
echo "  - data/splits/aux.json (10k indices for QR training)"
echo "  - data/splits/eval_in.json (5k positive samples)"
echo "  - data/splits/eval_out.json (5k negative samples)"

