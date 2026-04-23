#!/bin/bash
# Stage 3b/3 — Merge per-shard augment outputs into the final
# metadata_ligval.parquet. Run after augment_array.sbatch completes (32 shard
# parquets expected in ${DATASET_DIR}/ligand_validity_augment_shards/).
#
# Tip: chain via dependency:
#   ARR=$(sbatch --parsable augment_array.sbatch)
#   sbatch --dependency=afterok:${ARR} finalize.sbatch

#SBATCH --job-name=ligval_merge
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --time=1:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_merge_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_merge_%j.err

set -euo pipefail

DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
BASE_DIR=/home/users/zhkim216/code/allatom-design
NUM_SHARDS=32

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

echo "[merge] dataset_dir=${DATASET_DIR}, num_shards=${NUM_SHARDS}"

python3 -m allatom_design.data.preprocessing.atomworks.ligand_validity.augment_merge \
    --metadata-in  "${DATASET_DIR}/metadata.parquet" \
    --shard-dir    "${DATASET_DIR}/ligand_validity_augment_shards" \
    --metadata-out "${DATASET_DIR}/metadata_ligval.parquet" \
    --num-shards   "${NUM_SHARDS}"
