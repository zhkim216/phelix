#!/bin/bash
#SBATCH --job-name=eval
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

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

cd ${BASE_DIR}

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Denoiser path
MODEL_NAME=s3_a0
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Input config
PHASE=eval
PDB_DIR=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v1/esmfold_preds
PDB_NAME_LIST=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v1/${PHASE}_pdb_names.list

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_scaffold_training_${PHASE}
GROUP=eval_scaffold_training

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_scaffold_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    seq_des_cfg.fampnn.fampnn_ckpt=${FAMPNN_CKPT} \
    seq_des_cfg.fampnn.overrides.noise_labels=${NOISE_LABELS} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    input_cfg.subset_length_range=[32,256] \
    input_cfg.n_subsample=128 \
    start_step=0 \
    eval_every_n_ckpts=4 \
    hydra.run.dir=${HYDRA_DIR}
