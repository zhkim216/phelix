#!/bin/bash
# ==============================================================================
# Merge shard CSVs from array job into final CSVs
#
# Usage:
#   bash scripts/sherlock_scripts/jinho/eval_seq_des_training/merge_scaffold_constraint_shards.sh <output_dir>
#
# Example:
#   bash merge_scaffold_constraint_shards.sh /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/scaffold_pos_constraint_mad
# ==============================================================================

OUTPUT_DIR=${1:?Usage: $0 <output_dir>}

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "Error: directory ${OUTPUT_DIR} does not exist"
    exit 1
fi

K_VALUES=("02" "06" "10" "14" "18")

for K in "${K_VALUES[@]}"; do
    # Merge minimal CSVs
    MINIMAL_SHARDS=(${OUTPUT_DIR}/scaffold_constraint_mad_k${K}_shard*[0-9].csv)
    if [ ${#MINIMAL_SHARDS[@]} -eq 0 ] || [ ! -f "${MINIMAL_SHARDS[0]}" ]; then
        echo "Warning: no shard files found for k=${K} (minimal)"
        continue
    fi

    MERGED_MINIMAL=${OUTPUT_DIR}/scaffold_constraint_mad_k${K}.csv
    # Header from first shard, then data from all shards
    head -1 "${MINIMAL_SHARDS[0]}" > "${MERGED_MINIMAL}"
    for SHARD in "${MINIMAL_SHARDS[@]}"; do
        tail -n +2 "${SHARD}" >> "${MERGED_MINIMAL}"
    done
    ROWS=$(( $(wc -l < "${MERGED_MINIMAL}") - 1 ))
    echo "k=${K}: merged ${#MINIMAL_SHARDS[@]} shards -> ${MERGED_MINIMAL} (${ROWS} rows)"

    # Merge full CSVs
    FULL_SHARDS=(${OUTPUT_DIR}/scaffold_constraint_mad_k${K}_shard*_full.csv)
    if [ ${#FULL_SHARDS[@]} -eq 0 ] || [ ! -f "${FULL_SHARDS[0]}" ]; then
        echo "Warning: no shard files found for k=${K} (full)"
        continue
    fi

    MERGED_FULL=${OUTPUT_DIR}/scaffold_constraint_mad_k${K}_full.csv
    head -1 "${FULL_SHARDS[0]}" > "${MERGED_FULL}"
    for SHARD in "${FULL_SHARDS[@]}"; do
        tail -n +2 "${SHARD}" >> "${MERGED_FULL}"
    done
    ROWS=$(( $(wc -l < "${MERGED_FULL}") - 1 ))
    echo "k=${K}: merged ${#FULL_SHARDS[@]} shards -> ${MERGED_FULL} (${ROWS} rows)"
done

echo ""
echo "Merge complete. Final CSVs in ${OUTPUT_DIR}/"
echo "You can now delete shard files with:"
echo "  rm ${OUTPUT_DIR}/*_shard*.csv"
