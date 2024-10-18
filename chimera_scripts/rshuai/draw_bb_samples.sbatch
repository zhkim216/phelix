#!/bin/bash
#SBATCH --job-name=a13_bb0_draw_samples
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
# NNTM_DATASET=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/pdb_store_renumbered
NNTM_DATASET=null  # null to disable nntm computation
DSSP_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/dssp/dssp_length_strict.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Denoiser checkpoint
MODEL_NAME=a13_bb0
AD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/ad-step650000-epoch3915.ckpt

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_bb_samp_noaug
GROUP=${MODEL_NAME}

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/draw_bb_samples.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    ad_ckpt=${AD_CKPT} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    ss_kld.dssp_csv=${DSSP_CSV} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    n_samples_per_length=10 \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
