#!/bin/bash
# Stage 2/3 — Consolidate the per-PDB JSON cache (left behind by --no-consolidate
# fetch array tasks) into a single ligand_validity_cache.parquet. One-time prereq
# for the augment array job.
#
# Tip: this is pure I/O (~5-15 min on Sherlock) and trivially fits on an
# interactive node — feel free to skip the sbatch and run the python module
# directly via sh_dev / srun if the queue is slow.

#SBATCH --job-name=ligval_consolidate
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=1:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_consolidate_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_consolidate_%j.err

set -euo pipefail

DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
BASE_DIR=/home/users/zhkim216/code/allatom-design

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

echo "[consolidate] dataset_dir=${DATASET_DIR}"

python3 -m allatom_design.data.preprocessing.atomworks.ligand_validity.consolidate_cache \
    --cache-dir     "${DATASET_DIR}/ligand_validity_cache_json" \
    --cache-parquet "${DATASET_DIR}/ligand_validity_cache.parquet"
