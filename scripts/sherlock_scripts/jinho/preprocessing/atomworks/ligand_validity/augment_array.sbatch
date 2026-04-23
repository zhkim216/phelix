#!/bin/bash
# Stage 3a/3 — Run the augment step in parallel across 32 SLURM array tasks.
# Each task processes pdb_ids[shard_id::num_shards] (matches the fetch sharding
# convention) and writes its (row_idx, q_pn_unit_ligand_validity) pairs to
# `${DATASET_DIR}/ligand_validity_augment_shards/augment_shard_{id:05d}.parquet`.
#
# Prereq: ${DATASET_DIR}/ligand_validity_cache.parquet must exist
#         (run consolidate_cache.sbatch first if it doesn't).

#SBATCH --job-name=ligval_augment
#SBATCH -p possu,normal,owners
#SBATCH --array=0-31
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=2:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_augment_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_augment_%A_%a.err

set -euo pipefail

DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
BASE_DIR=/home/users/zhkim216/code/allatom-design
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX + 1))

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

echo "[augment shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}] dataset_dir=${DATASET_DIR}"

python3 -m allatom_design.data.preprocessing.atomworks.ligand_validity.augment_shard \
    --metadata-in   "${DATASET_DIR}/metadata.parquet" \
    --cache-parquet "${DATASET_DIR}/ligand_validity_cache.parquet" \
    --shard-out-dir "${DATASET_DIR}/ligand_validity_augment_shards" \
    --shard-id      "${SLURM_ARRAY_TASK_ID}" \
    --num-shards    "${NUM_SHARDS}"
