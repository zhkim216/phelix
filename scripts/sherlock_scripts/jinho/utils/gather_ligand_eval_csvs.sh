#!/bin/bash
# Edit EXP_DIRS and OUTPUT_TAR below, then run: bash gather_ligand_eval_csvs.sh
#
# Gathers CSV outputs produced by `allatom_design.eval.glide.run_ligand_eval_batch`
# (ligand eval batch). Set ARRAY_JOBS=true to also collect per-SLURM-array shards
# (e.g. ligand_eval_metrics_lplddt80_lrmsd2_array_3.csv).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp37_cfg11_denovoval_twostage_ps_ligand_eval
)

OUTPUT_TAR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/collected_csvs/exp37_cfg11_denovoval_twostage_ps_ligand_eval.tar.gz

# Add --array-jobs flag to also collect *_array_N.csv files
ARRAY_JOBS=true

if [ "$ARRAY_JOBS" = true ]; then
    python3 "$SCRIPT_DIR/gather_ligand_eval_csvs.py" --array-jobs "$OUTPUT_TAR" "${EXP_DIRS[@]}"
else
    python3 "$SCRIPT_DIR/gather_ligand_eval_csvs.py" "$OUTPUT_TAR" "${EXP_DIRS[@]}"
fi
