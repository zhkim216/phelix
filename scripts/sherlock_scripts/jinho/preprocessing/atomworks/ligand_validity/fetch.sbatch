#!/bin/bash
# Stage 1/3 — Fetch RCSB ligand-validity JSON per PDB into the per-PDB cache
# directory. Resume-safe: --skip-existing (default) skips already-fetched PDBs.
# Sharded: each array task fetches pdb_ids[task_id::num_tasks].
#
# After all tasks complete, run consolidate_cache.sbatch (or its interactive
# equivalent) to roll the JSON cache into a single ligand_validity_cache.parquet
# that the augment array can read.

#SBATCH --job-name=ligval_fetch
#SBATCH -p possu,normal,owners
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_fetch_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_fetch_%A_%a.err

set -euo pipefail

DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
BASE_DIR=/home/users/zhkim216/code/allatom-design
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX + 1))

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

echo "[fetch shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}] dataset_dir=${DATASET_DIR}"

python3 -m allatom_design.data.preprocessing.atomworks.ligand_validity.fetch \
    --metadata-in   "${DATASET_DIR}/metadata.parquet" \
    --cache-dir     "${DATASET_DIR}/ligand_validity_cache_json" \
    --cache-parquet "${DATASET_DIR}/ligand_validity_cache.parquet" \
    --num-workers   8 \
    --shard-id      "${SLURM_ARRAY_TASK_ID}" \
    --num-shards    "${NUM_SHARDS}" \
    --no-consolidate
