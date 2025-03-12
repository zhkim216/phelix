#!/bin/bash
#SBATCH --job-name=if_eval
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0
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
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset

# Define models array
models=(
c1_aasd3_eps0.3
)

# Get model name from array
MODEL_NAME=${models[$SLURM_ARRAY_TASK_ID]}

EXP_NAME=${MODEL_NAME}_eval_scn_pack
GROUP=c1_aasd2_eps_sweep

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_scn_pack_training.py \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    data.pdb_path=${PDB_PATH} \
    subset_length_range=[64,512] \
    hydra.run.dir=${HYDRA_DIR}
