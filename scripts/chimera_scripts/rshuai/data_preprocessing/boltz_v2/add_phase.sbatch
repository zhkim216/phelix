#!/bin/bash
#SBATCH --job-name=pretokenize_inputs
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
BASE_DIR=/home/talal/rshuai/allatom-design

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/boltz_v2/add_phase.py \
    pdb_path=/scratch/hielab/rshuai/datasets/boltz_v2 \
    num_workers=64