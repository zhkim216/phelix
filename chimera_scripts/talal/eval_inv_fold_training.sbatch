#!/bin/bash
#SBATCH --job-name=eval_fitness
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err
#

# Set base directories
BASE_DIR=/home/talal/allatom-design
HYDRA_DIR=/scratch/hielab/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/casp14

# Set cache directories
export HF_HOME=/scratch/hielab/talal/cache/huggingface
export TORCH_HOME=/scratch/hielab/talal/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/talal/cache/omegafold/omegafold.pt

# Model to evaluate
MODEL_NAME=b8_aasd3
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=talal-widatalla
EXP_NAME=${MODEL_NAME}_eval_inv_fold
GROUP=chimera_a7

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_inv_fold_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    run_seq_recovery=true \
    run_codes_sc=false \
    no_wandb=true \
    num_codes_sc_batches=1 \
    data.pdb_path=${PDB_PATH} \
    hydra.run.dir=${HYDRA_DIR}