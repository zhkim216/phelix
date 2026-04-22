#!/bin/bash
# Fetch RCSB ligand validity JSON per PDB (one array task per shard).
# Resume-safe: --skip-existing (Python default) skips already-fetched PDBs.
#
# ------------------------------------------------------------------
# 1) Fetch (submit via wrap_sbatch — array spec is baked in via #SBATCH below):
#      bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container.sh \
#          scripts/sherlock_scripts/jinho/preprocessing/atomworks/fetch_and_augment_ligand_validity.sbatch
#
# 2) After fetch completes, consolidate JSON cache + join to metadata
#    (run inside container, e.g. via srun or a simple sbatch):
#      python3 -m allatom_design.data.preprocessing.atomworks.fetch_and_augment_ligand_validity \
#          --metadata-in   ${DATASET_DIR}/metadata.parquet \
#          --metadata-out  ${DATASET_DIR}/metadata_ligval.parquet \
#          --cache-dir     ${DATASET_DIR}/ligand_validity_cache_json \
#          --cache-parquet ${DATASET_DIR}/ligand_validity_cache.parquet \
#          --only-augment
# ------------------------------------------------------------------

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

python3 -m allatom_design.data.preprocessing.atomworks.fetch_and_augment_ligand_validity \
    --metadata-in   "${DATASET_DIR}/metadata.parquet" \
    --metadata-out  "${DATASET_DIR}/metadata_ligval.parquet" \
    --cache-dir     "${DATASET_DIR}/ligand_validity_cache_json" \
    --cache-parquet "${DATASET_DIR}/ligand_validity_cache.parquet" \
    --num-workers   8 \
    --shard-id      "${SLURM_ARRAY_TASK_ID}" \
    --num-shards    "${NUM_SHARDS}" \
    --no-consolidate \
    --only-fetch
