#!/bin/bash
#SBATCH --job-name=a8_aasd0_eval_inv_fold
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh

conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset

# Model name to evaluate
MODEL_NAME=a8_aasd0_eps_0.0

EXP_NAME=${MODEL_NAME}_eval_inv_fold
GROUP=chimera_a8

# Set HF transformers cache
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_inv_fold_training.py \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    num_steps_list=[100] \
    run_seq_recovery=false \
    run_codes_sc=true \
    num_codes_sc_batches=1 \
    data.pdb_path=${PDB_PATH} \
    subset_length_range=[64,256] \
    hydra.run.dir=${HYDRA_DIR}