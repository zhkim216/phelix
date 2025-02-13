#!/bin/bash
#SBATCH --job-name=c3_bb0_eval_fpd_ema
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-9%3
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.subhang_bash_profile
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Set base directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Dataset paths
FPD_EMBEDDINGS_DIR=/scratch/users/rshuai/datasets/ingraham_cath_dataset/fpd_embeddings

# Set HF variables
export HUGGINGFACE_TOKEN=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
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Denoiser checkpoint
MODEL_NAME=c3_bb0
CKPT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}/post_hoc_ema_ckpts

# Wandb settings
WANDB_ID=null
GROUP=eval_fpd

# Define step size and maximum step to sweep over
STEP_SIZE=0.01
MAX_STEP=0.10
current_step=$(echo "scale=2; $STEP_SIZE * $SLURM_ARRAY_TASK_ID" | bc)
formatted_step=$(printf "%.3f" "$current_step")
MODEL_CKPT="${CKPT_DIR}/ema-step500000-std${formatted_step}.ckpt"

# Set the experiment name to include the current MODEL_CKPT step
EXP_NAME=${MODEL_NAME}_fpd_eval_ema${formatted_step}

# Optionally, set a group name that reflects the checkpoint step
GROUP=${MODEL_NAME}_fpd_ema

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_fpd.py \
    hydra.run.dir=${HYDRA_DIR} \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    model_ckpt=${MODEL_CKPT} \
    fpd_embeddings_dir=${FPD_EMBEDDINGS_DIR}
