#!/bin/bash
#SBATCH --job-name=ligclust_v8
#SBATCH -p possu,normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/ligclust_v8_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/ligclust_v8_%j.err

set -euo pipefail

BASE_DIR=/home/users/zhkim216/code/allatom-design
DATASET_DIR=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8
BL_DIR="${BASE_DIR}/allatom_design/data/preprocessing/bl_ligand_curation"

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd "${BASE_DIR}"

# Inputs:
#   ${DATASET_DIR}/metadata_ligval_seq_clustered_04_bmsm.parquet
#     — produced by augment_metadata_with_bmsm (step2 of v8_pipeline.sh)
#   ${BL_DIR}/ccd_smiles_cache_metadata_v8.json
#     — 44,364 CCD → SMILES, includes 273 plinder artifacts
#
# Output:
#   ${DATASET_DIR}/metadata_ligval_seq_clustered_04_bmsm_ligclust.parquet
#     — adds q_pn_unit_bmsm_ligand_cluster_id (int32, -1 for non-BMSM rows)
#
# Algorithm: complete-linkage hierarchical clustering on Morgan ECFP4
# (radius=2, nBits=2048) with Tanimoto distance. Cluster diameter is strictly
# bounded by (1 - tanimoto_cutoff), so every intra-cluster pair is guaranteed
# Tanimoto >= cutoff. Replaces previous Butina implementation which allowed
# chaining up to 1.67x the nominal cutoff (see debug/260417 README).
#
# Resource notes:
#   - BMSM subset has ~43K unique canonical SMILES -> ~0.93B pairwise distances
#     (~7.4 GB float64 condensed). fastcluster peak ~8 GB + fingerprints.
#   - 32 GB mem gives ~4x headroom; 4 h is ~8-12x the expected wall time.
#   - Single-threaded: fastcluster NN-chain and RDKit BulkTanimotoSimilarity
#     do not benefit from extra CPUs here.

python3 -m allatom_design.data.preprocessing.atomworks.cluster_ligand_butina \
    --input-parquet     "${DATASET_DIR}/metadata_ligval_seq_clustered_04_bmsm.parquet" \
    --output-parquet    "${DATASET_DIR}/metadata_ligval_seq_clustered_04_bmsm_ligclust.parquet" \
    --smiles-cache-json "${BL_DIR}/ccd_smiles_cache_metadata_v8.json" \
    --tanimoto-cutoff   0.8 \
    --morgan-radius     2 \
    --morgan-bits       2048
