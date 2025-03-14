#!/bin/bash
#SBATCH --job-name=eval_ai_dit_12b_ema99_fampnn
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
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn

# FAMPNN options
FAMPNN_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt
NOISE_LABELS=0.0

# Set HF transformers cache
export HF_HOME=/scratch/users/rshuai/cache/huggingface

cd ${BASE_DIR}

# Dataset paths
NNTM_DATASET=null

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/users/rshuai/cache/omegafold/omegafold.pt

# Denoiser path
MODEL_NAME=ai_dit_12b_ema99
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_bb_gen_nl${NOISE_LABELS}
GROUP=ai_cath

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_bb_gen_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=8 \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    seq_des_cfg.fampnn.fampnn_ckpt=${FAMPNN_CKPT} \
    seq_des_cfg.fampnn.overrides.noise_labels=${NOISE_LABELS} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    nntm_dataset=${NNTM_DATASET} \
    length_range=[32,256] \
    length_step_size=4 \
    n_samples_per_length=8 \
    hydra.run.dir=${HYDRA_DIR}
