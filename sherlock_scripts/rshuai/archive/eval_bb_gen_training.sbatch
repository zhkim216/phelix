#!/bin/bash
#SBATCH --job-name=s4_a3_eval_bb_gen
#SBATCH -p bioe
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
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
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

# Denoiser path
MODEL_NAME=s4_a3
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_bb_gen
GROUP=backbone

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_bb_gen_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    seq_des_cfg.fampnn.fampnn_ckpt=${FAMPNN_CKPT} \
    seq_des_cfg.fampnn.overrides.noise_labels=${NOISE_LABELS} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    start_step=0 \
    eval_every_n_ckpts=2 \
    hydra.run.dir=${HYDRA_DIR}
