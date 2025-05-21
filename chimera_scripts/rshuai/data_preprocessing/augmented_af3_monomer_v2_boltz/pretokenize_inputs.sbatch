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
python3 allatom_design/data/preprocessing/augmented_af3_monomer_v2_boltz/pretokenize_inputs.py \
    pdb_path=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v2_boltz \
    num_workers=16