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
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg0_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg1_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg3_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg4_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg5_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg6_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg7_denovoval_designable_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg8_denovoval_designable_af3
    )
    OUTPUT_TAR="$BASE_DIR/collected_csvs/exp35_denovo_designable_sweep.tar.gz"
fi

echo "Output: $OUTPUT_TAR"

if [ "$ARRAY_JOBS" = true ]; then
    python3 "$SCRIPT_DIR/gather_csvs.py" --array-jobs "$OUTPUT_TAR" "${EXP_DIRS[@]}"
else
    python3 "$SCRIPT_DIR/gather_csvs.py" "$OUTPUT_TAR" "${EXP_DIRS[@]}"
fi
