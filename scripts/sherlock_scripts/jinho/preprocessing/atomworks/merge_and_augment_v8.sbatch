#!/bin/bash
#SBATCH --job-name=merge_augment_v8
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/merge_augment_v8_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/merge_augment_v8_%j.err

set -euo pipefail

BASE_DIR=/home/users/zhkim216/code/allatom-design
DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd ${BASE_DIR}

echo "=== Step 1: Merge parquet shards ==="
python3 allatom_design/data/preprocessing/atomworks/merge_parquet_shards.py \
    --out_dir "${DATASET_DIR}"

echo "=== Step 2: Augment with physical annotations ==="
python3 -m allatom_design.data.preprocessing.atomworks.augment_metadata_with_physical_annotations \
    --input-parquet "${DATASET_DIR}/metadata.parquet" \
    --output-parquet "${DATASET_DIR}/metadata_augmented.parquet" \
    --min-metal-occupancy 0.5 \
    --min-metal-coordination-partners 3 \
    --min-small-molecule-neighboring-heavy-atoms 3 \
    --context-distance 5.0

echo "=== Step 3: Exclude CCD codes (NA, K, CL, BR) ==="
python3 -m allatom_design.data.preprocessing.atomworks.exclude_ccd_codes_from_metadata \
    --input-parquet "${DATASET_DIR}/metadata_augmented.parquet" \
    --output-parquet "${DATASET_DIR}/metadata_augmented_lmpnn.parquet" \
    --deleted-chains-out "${DATASET_DIR}/metadata_augmented_deleted_chains.parquet" \
    --exclude-ccd-codes NA K CL BR

echo "=== Done ==="
echo "Outputs:"
echo "  Merged:    ${DATASET_DIR}/metadata.parquet"
echo "  Augmented: ${DATASET_DIR}/metadata_augmented.parquet"
echo "  Final:     ${DATASET_DIR}/metadata_augmented_lmpnn.parquet"
