#!/bin/bash
#SBATCH --job-name=eval_val_ema
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-28%5
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Checkpoint directory
MODEL_NAME=b13_aasd6
CKPT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/post_hoc_ema_ckpts

# Define EMA steps to sweep over
EMA_STEPS=( $(seq 0.01 0.01 0.295) )
ema_step=${EMA_STEPS[${SLURM_ARRAY_TASK_ID}]}
ema_step=$(printf "%.3f" "$ema_step")
SD_CKPT="${CKPT_DIR}/ema-step150000-std${ema_step}.ckpt"

# Wandb settings
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_ema${ema_step}
GROUP=${MODEL_NAME}_ema

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_validation.py \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    sd_ckpt=${SD_CKPT} \
    batch_size=16 \
    data.subset_length_range=[0,9999] \
    hydra.run.dir=${HYDRA_DIR}
