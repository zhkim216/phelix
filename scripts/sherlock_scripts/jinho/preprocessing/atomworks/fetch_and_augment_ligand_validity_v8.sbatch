#!/bin/bash
#SBATCH --job-name=ligval_v8
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_v8_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligval_v8_%j.err

set -euo pipefail

BASE_DIR=/home/users/zhkim216/code/allatom-design
DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

# Inputs produced by merge_and_augment_v8.sbatch:
#   ${DATASET_DIR}/metadata_augmented_lmpnn.parquet   (post-CCD-exclusion metadata)
#
# Outputs:
#   ${DATASET_DIR}/ligand_validity_cache_json/{pdb[:2]}/{pdb}.json  (per-PDB cache, resume-friendly)
#   ${DATASET_DIR}/ligand_validity_cache.parquet                    (consolidated cache)
#   ${DATASET_DIR}/metadata_augmented_lmpnn_ligval.parquet          (final metadata)
#
# Re-running this sbatch is safe: --skip-existing means already-fetched PDBs
# do not re-hit the RCSB network. Only failed-or-new PDBs are retried.

python3 -m allatom_design.data.preprocessing.atomworks.fetch_and_augment_ligand_validity \
    --metadata-in   "${DATASET_DIR}/metadata_augmented_lmpnn.parquet" \
    --metadata-out  "${DATASET_DIR}/metadata_augmented_lmpnn_ligval.parquet" \
    --cache-dir     "${DATASET_DIR}/ligand_validity_cache_json" \
    --cache-parquet "${DATASET_DIR}/ligand_validity_cache.parquet" \
    --num-workers   8 \
    --checkpoint-every 1000
