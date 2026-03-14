#!/bin/bash
# Edit EXP_DIRS below, then run: bash run_detect.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp37_cfg0_nativeval_sm_selected_seq4_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp37_cfg1_nativeval_sm_selected_seq4_af3
)

for exp_dir in "${EXP_DIRS[@]}"; do
    echo "=========================================="
    echo "Processing: $exp_dir"
    echo "=========================================="
    python3 "$SCRIPT_DIR/detect_incomplete.py" "$exp_dir" --output-dir "$SCRIPT_DIR"
    echo ""
done
