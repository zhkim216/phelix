#!/bin/bash
#SBATCH --job-name=phema
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
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

cd ${BASE_DIR}

# Model name to evaluate
MODEL_NAME=a10_aasd3
MODEL_TYPE=seq_denoiser  # atom_denoiser or seq_denoiser

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/reconstruct_phema.py \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    model_type=${MODEL_TYPE} \
    out_step=200000 \
    out_stds=null \
    hydra.run.dir=${HYDRA_DIR}
