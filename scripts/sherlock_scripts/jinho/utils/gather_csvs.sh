#!/bin/bash
# Edit EXP_DIRS and OUTPUT_TAR below, then run: bash gather_csvs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp37_cfg11_denovoval_seq8_af3_ccd0VI_len150_250_350_450    
)

OUTPUT_TAR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/collected_csvs/denovoval_scaffold_generation_remaining_test.tar.gz

# Add --array-jobs flag to also collect *_array_N.csv files
ARRAY_JOBS=false

if [ "$ARRAY_JOBS" = true ]; then
    python3 "$SCRIPT_DIR/gather_csvs.py" --array-jobs "$OUTPUT_TAR" "${EXP_DIRS[@]}"
else
    python3 "$SCRIPT_DIR/gather_csvs.py" "$OUTPUT_TAR" "${EXP_DIRS[@]}"
fi
