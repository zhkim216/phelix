#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== Edit here: add experiment directories =====
EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp36_cfg1_rfd3val_af3
)

for exp_dir in "${EXP_DIRS[@]}"; do
    echo "=========================================="
    echo "Processing: $exp_dir"
    echo "=========================================="
    python3 "$SCRIPT_DIR/detect_incomplete.py" "$exp_dir" --output-dir "$SCRIPT_DIR"
    echo ""
done
