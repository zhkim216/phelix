#!/bin/bash
#SBATCH --job-name=eval_seq_des
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
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
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
SOFTWARE_PATH=/scratch/hielab/rshuai/software
# SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software  # e.g. for MASTER, redis

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Inputs config
PDB_DIR=/scratch/hielab/rshuai/datasets/denovo250dev/pdbs

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Model to evaluate
MODEL_NAME=sd1_a1_cfg0
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_seq_des_denovo250dev_L300
GROUP=seq_des

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_seq_des_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    software_path=${SOFTWARE_PATH} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.model_name=af2 \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.subset_length_range=[300,300] \
    start_step=0 \
    end_step=null \
    eval_every_n_ckpts=1 \
    hydra.run.dir=${HYDRA_DIR}
