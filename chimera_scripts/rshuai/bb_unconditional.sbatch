#!/bin/bash
#SBATCH --job-name=bb_unconditional
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Define directories and paths
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Sequence design model params
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
FAMPNN_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt
NOISE_LABELS=0.0

# Structure prediction model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Dataset paths for certain eval metrics
NNTM_DATASET=null  # null to disable nntm computation

# Denoiser checkpoint
CHECKPOINT_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/u1_a7/checkpoints/ema/ad-step1625000-epoch2008-ema0.99.ckpt
BASE_OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/u1_a7/bb_unconditional

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=u1_a7_bb_unconditional
GROUP=bb_unconditional

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/bb_unconditional.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    checkpoint_path=${CHECKPOINT_PATH} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    seq_des_cfg.fampnn.fampnn_ckpt=${FAMPNN_CKPT} \
    seq_des_cfg.fampnn.overrides.noise_labels=${NOISE_LABELS} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    hydra.run.dir=${HYDRA_DIR}
