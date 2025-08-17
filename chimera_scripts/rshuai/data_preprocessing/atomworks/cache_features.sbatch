#!/bin/bash
#SBATCH --job-name=cache_features
#SBATCH -p cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
ENV_DIR=/scratch/hielab/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

export PDB_MIRROR_PATH=""
export CCD_MIRROR_PATH="/scratch/hielab/rshuai/datasets/ccd"
BASE_DIR=/home/talal/rshuai/allatom-design

cd ${BASE_DIR}
# Run script
python3 allatom_design/data/preprocessing/atomworks/cache_features.py \
    mmcif_dir=/large_storage/hielab/rshuai/datasets/mmCIF \
    out_dir=/scratch/hielab/rshuai/datasets/atomworks \
    num_workers=64
