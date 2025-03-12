#!/bin/bash
#SBATCH --job-name=eval_fitness
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=97:59:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/talal/slurm_log_files/job_$SLURM_JOB_ID.sh

# Set base directories
BASE_DIR=/home/talal/allatom-design
HYDRA_DIR=/scratch/hielab/talal/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/talal/cache/huggingface
export TORCH_HOME=/scratch/hielab/talal/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/talal/cache/omegafold/omegafold.pt

# Model to evaluate
MODEL_NAME=b12_aasd6
DENOISER_TRAIN_DIR=/scratch/hielab/talal/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
EXP_NAME=${MODEL_NAME}_fitness
WANDB_ID=talal-widatalla
GROUP=fitness_eval

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_fitness_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    no_wandb=false \
    checkpoint_override=/scratch/hielab/rshuai/allatom_design/allatom_design/b12_aasd6/checkpoints/sd-step580000-epoch345.ckpt \
    group=${GROUP} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    start_epoch=0 \
    eval_every_n_ckpts=1 \
    batch_size=8\
    datasets=[megascale_test,s669,skempi] \
    hydra.run.dir=${HYDRA_DIR} \
    data.overwrite_cache=false \
