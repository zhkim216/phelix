#!/bin/bash
# Merge array CSV files for all lc_seq_des_multi experiments
# Usage: bash merge_all_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg0_denovoval_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg1_denovoval_af3
)

echo "=============================================="
echo "Merging CSV files for ${#EXP_DIRS[@]} experiments"
echo "=============================================="

for EXP_DIR in "${EXP_DIRS[@]}"; do
    if [ -d "${EXP_DIR}" ]; then
        echo ""
        echo "Processing: $(basename ${EXP_DIR})"
        python3 ${SCRIPT_DIR}/merge_array_csvs.py \
            --results_dir "${EXP_DIR}"
    else
        echo "SKIP (not found): $(basename ${EXP_DIR})"
    fi
done

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
