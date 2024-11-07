#!/bin/bash
#SBATCH --job-name=b3_aasd5_inv_fold
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-14%5
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

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
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Sweep parameters
temperature_values=(0.0 0.1 0.2)
num_steps_values=(1 10 20 100 200)
tau=${temperature_values[$((SLURM_ARRAY_TASK_ID / 5))]}
S=${num_steps_values[$((SLURM_ARRAY_TASK_ID % 5))]}

# Denoiser checkpoint
MODEL_NAME=b3_aasd5
SD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/sd-step200000-epoch600.ckpt

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_inv_fold_tau${tau}_S${S}
GROUP=${MODEL_NAME}_inv_fold

# Run inverse folding script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/inverse_fold.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    sd_ckpt=${SD_CKPT} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    temperature=${tau} \
    timestep_schedule.num_steps=${S} \
    num_pdbs=300 \
    num_seqs_per_pdb=1 \
    batch_size=64 \
    hydra.run.dir=${HYDRA_DIR}
