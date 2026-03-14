#!/bin/bash
# Usage: bash run_detect.sh /path/to/exp_dir1 /path/to/exp_dir2 ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: bash run_detect.sh <experiment_dir1> [experiment_dir2] ..."
    exit 1
fi

for exp_dir in "$@"; do
    echo "=========================================="
    echo "Processing: $exp_dir"
    echo "=========================================="
    python3 "$SCRIPT_DIR/detect_incomplete.py" "$exp_dir" --output-dir "$SCRIPT_DIR"
    echo ""
done
