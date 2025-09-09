#!/bin/bash
#SBATCH --job-name=cluster_sequences
#SBATCH -p bioe
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ENV_DIR=/scratch/users/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

export PDB_MIRROR_PATH=""
export CCD_MIRROR_PATH="/scratch/users/rshuai/datasets/ccd"
export SOFTWARE_PATH=/scratch/users/rshuai/software

BASE_DIR=/home/users/rshuai/code/allatom-design

# Run script
cd ${BASE_DIR}
python3 allatom_design/data/preprocessing/atomworks/cluster_sequences.py \
    pdb_path=/scratch/users/rshuai/datasets/atomworks
