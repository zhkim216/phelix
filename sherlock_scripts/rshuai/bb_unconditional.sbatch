#!/bin/bash
#SBATCH --job-name=bb_unconditional
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.1
ENV_DIR=/oak/stanford/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Sequence design model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
FAMPNN_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt
NOISE_LABELS=0.0

# Structure prediction model params
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Dataset paths for certain eval metrics
NNTM_DATASET=null  # null to disable nntm computation

# Denoiser checkpoint
CHECKPOINT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/u1_a7/checkpoints/ema/ad-step1625000-epoch2008-ema0.99.ckpt
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/u1_a7/bb_unconditional

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=u1_a7_bb_unconditional_sherlock
GROUP=backbone

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
