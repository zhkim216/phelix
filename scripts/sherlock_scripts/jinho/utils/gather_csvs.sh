#!/bin/bash
# Usage:
#   bash gather_csvs.sh                        # Use hardcoded EXP_DIRS
#   bash gather_csvs.sh --pattern <glob>       # Auto-discover dirs matching pattern
#
# Examples:
#   bash gather_csvs.sh --pattern "eval_exp37_cfg11_denovoval_seq8_af3*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
BASE_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
ARRAY_JOBS=false

# --- Parse arguments ---
PATTERN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pattern) PATTERN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -n "$PATTERN" ]; then
    # Auto-discover directories matching the pattern
    EXP_DIRS=()
    for d in "$BASE_DIR"/$PATTERN; do
        [ -d "$d" ] && EXP_DIRS+=("$d")
    done
    if [ ${#EXP_DIRS[@]} -eq 0 ]; then
        echo "No directories found matching: $BASE_DIR/$PATTERN"
        exit 1
    fi
    # Derive output tar name from pattern (strip trailing *)
    PATTERN_STEM="${PATTERN%%\*}"
    PATTERN_STEM="${PATTERN_STEM%_}"
    OUTPUT_TAR="$BASE_DIR/collected_csvs/${PATTERN_STEM}.tar.gz"
    echo "Found ${#EXP_DIRS[@]} directories matching '$PATTERN':"
    printf '  %s\n' "${EXP_DIRS[@]}"
else
    # Hardcoded fallback
    EXP_DIRS=(
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg0_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg2_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg4_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg5_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg7_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg9_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg10_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg12_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp22_cfg14_nativeval_sm_selected_seq4_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_deep_potts_pocket_015_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_deep_potts_pocket_020_af3
    # /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_deep_potts_pocket_025_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg0_denovoval_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg1_denovoval_af3
    )
    OUTPUT_TAR="$BASE_DIR/collected_csvs/exp41_denovoval_len150250_sweep.tar.gz"
fi

echo "Output: $OUTPUT_TAR"

if [ "$ARRAY_JOBS" = true ]; then
    python3 "$SCRIPT_DIR/gather_csvs.py" --array-jobs "$OUTPUT_TAR" "${EXP_DIRS[@]}"
else
    python3 "$SCRIPT_DIR/gather_csvs.py" "$OUTPUT_TAR" "${EXP_DIRS[@]}"
fi
