#!/bin/bash
# Cluster BMSM ligands by Tanimoto on Morgan ECFP4.
# Adds q_pn_unit_bmsm_ligand_cluster_id (int32, -1 for non-BMSM rows).
#
# Submit:
#   bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container.sh \
#       scripts/sherlock_scripts/jinho/preprocessing/atomworks/cluster_ligand_butina.sbatch

#SBATCH --job-name=ligclust
#SBATCH -p possu,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligclust_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligclust_%j.err

set -euo pipefail

DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
SEQ_ID_TAG=04
VERSION_TAG=v9

BASE_DIR=/home/users/zhkim216/code/allatom-design
BL_DIR="${BASE_DIR}/allatom_design/data/preprocessing/bl_ligand_curation"
SMILES_CACHE="${BL_DIR}/ccd_smiles_cache_metadata_${VERSION_TAG}.json"
INPUT_PARQUET="${DATASET_DIR}/metadata_ligval_seq_clustered_${SEQ_ID_TAG}_bmsm.parquet"
OUTPUT_PARQUET="${DATASET_DIR}/metadata_ligval_seq_clustered_${SEQ_ID_TAG}_bmsm_ligclust.parquet"

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

python3 -m allatom_design.data.preprocessing.atomworks.cluster_ligand_butina \
    --input-parquet     "${INPUT_PARQUET}" \
    --output-parquet    "${OUTPUT_PARQUET}" \
    --smiles-cache-json "${SMILES_CACHE}" \
    --tanimoto-cutoff   0.8 \
    --morgan-radius     2 \
    --morgan-bits       2048
