#!/bin/bash
#SBATCH --job-name=test
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
WANDB_ID=richardshuai

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Wandb settings / experiment name
EXP_NAME=sd0_a0
GROUP=seq_des

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/boltz

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_path=${PDB_PATH} \
    train.batch_size=32 \
    trainer.accumulate_grad_batches=2 \
    train.compile_model=true \
    denoiser.per_residue_eps=false \
    denoiser.augment_eps=0.3 \
    checkpointing.save_latest_every_n_steps=5000 \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR}
