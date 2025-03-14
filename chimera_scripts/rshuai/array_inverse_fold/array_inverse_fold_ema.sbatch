#!/bin/bash
#SBATCH --job-name=a10_aasd3_inv_fold
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-57%10  # must be precomputed
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Denoiser checkpoint directory
MODEL_NAME=a10_aasd3
CKPT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/post_hoc_ema_ckpts

# Get the list of checkpoint files and sort them
CKPT_FILES=($(ls ${CKPT_DIR}/*.ckpt | sort))

# Get the checkpoint corresponding to the SLURM_ARRAY_TASK_ID
SD_CKPT=${CKPT_FILES[$SLURM_ARRAY_TASK_ID]}

# Extract the checkpoint filename without the path for naming
CKPT_NAME=$(basename ${SD_CKPT} .ckpt)
STD_VALUE=${CKPT_NAME##*-std}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_ema${STD_VALUE}_inv_fold
GROUP=${MODEL_NAME}_${MODEL_NAME}_s0

# Run inverse folding script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/inverse_fold.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    sd_ckpt=${SD_CKPT} \
    num_workers=8 \
    data.pdb_path=${PDB_PATH} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    timestep_schedule.num_steps=1 \
    scn_diffusion.noise_schedule.c=1.0 \
    num_pdbs=300 \
    num_seqs_per_pdb=1 \
    batch_size=64 \
    hydra.run.dir=${HYDRA_DIR}
