#!/bin/bash
#SBATCH --job-name=metadata_atomworks_pdb_full
#SBATCH -p possu,normal,owners
#SBATCH --array=0-49
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=32GB

# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/job_%A_%a.err

## Load modules and set environment variables:
BASE_DIR=/home/users/zhkim216/code/allatom-design

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"

# Create output directories if they don't exist
mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/atomworks/build_metadata_parquet_shards.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=build_metadata_parquet_shards \
    mmcif_dir=/scratch/users/zhkim216/datasets/pdb_mirror \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=100 \
    shard_id=${SLURM_ARRAY_TASK_ID} \
    max_file_size=20000000 \
    data_preprocessor_cfg.polymer_pn_unit_limit=1000
