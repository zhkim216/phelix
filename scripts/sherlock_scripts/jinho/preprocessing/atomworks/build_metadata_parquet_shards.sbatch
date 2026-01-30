#!/bin/bash
#SBATCH --job-name=metadata_atomworks_pdb_full_v5
#SBATCH -p normal,owners
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --mem=128GB

# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v5_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v5_%A_%a.err

## Load modules and set environment variables:
BASE_DIR=/home/users/zhkim216/code/allatom-design

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"

# Create output directories if they don't exist
mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing

cd ${BASE_DIR}

# Calculate num_shards from SLURM_ARRAY_TASK_MAX (extracted from --array directive)
# SLURM_ARRAY_TASK_MAX gives the max task ID, so num_shards = max + 1
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX + 1))

# Run script
python3 allatom_design/data/preprocessing/atomworks/build_metadata_parquet_shards.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=build_metadata_parquet_shards \
    mmcif_dir=/scratch/users/zhkim216/datasets/pdb_mirror \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v5 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=${NUM_SHARDS} \
    shard_id=${SLURM_ARRAY_TASK_ID} \
    max_file_size=5000000 \
    fetch_ligand_validity_scores=false \
    data_preprocessor_cfg.polymer_pn_unit_limit=500 \
    data_preprocessor_cfg.contact_distance=6 \
    data_preprocessor_cfg.from_rcsb=true \
    processing_timeout=600 \
    batch_size=200 \
    max_tasks_per_child=25
