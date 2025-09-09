#!/bin/bash
#SBATCH --job-name=annotate_pdb_keys
#SBATCH -p cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-99%10
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
PHASE=train
python3 allatom_design/data/preprocessing/af3_interface/annotate_pdb_keys.py \
    --train_keys_list /scratch/hielab/rshuai/datasets/af3_pdb_interface/train_pdb_keys.list \
    --eval_keys_list /scratch/hielab/rshuai/datasets/af3_pdb_interface/eval_pdb_keys.list \
    --n_eval2 2000 \
    --out_dir /scratch/hielab/rshuai/datasets/af3_pdb_interface/pdb_key_annotations \
    --dataset_dir /scratch/hielab/rshuai/datasets/af3_pdb_interface
