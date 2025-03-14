#!/bin/bash
#SBATCH --job-name=a10_aasd7_eval_scn_pack
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err

## Load modules and set environment variables:
BASE_DIR=/home/talal/allatom-design
WANDB_ID=talal-widatalla

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/talal/datasets/casp15

# Model name to evaluate
MODEL_NAME=aasd_gvp_pdb

EXP_NAME=${MODEL_NAME}_eval_scn_pack
GROUP=chimera_a10

# Set HF transformers cache
export HF_HOME=/scratch/hielab/talal/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/hielab/talal/allatom_design/allatom_design/${MODEL_NAME}

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_scn_pack_training.py \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    wandb_id=${WANDB_ID} \
    num_workers=8 \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    data.pdb_path=${PDB_PATH} \
    hydra.run.dir=${HYDRA_DIR} \
    no_wandb=false \
