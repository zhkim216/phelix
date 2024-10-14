#!/bin/bash
#SBATCH --job-name=a12_se3
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
#SBATCH --array=0-7

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh

# Load modules and activate conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model parameters
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
NNTM_DATASET=null  # null to disable nntm computation
DSSP_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/dssp/dssp_length_strict.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Sampling sweep values
case $SLURM_ARRAY_TASK_ID in
  0)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=false
    translation_scale=0.0
    ;;
  1)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=true
    translation_scale=0.0
    ;;
  2)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=true
    translation_scale=1.0
    ;;
  3)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=true
    translation_scale=2.0
    ;;
  4)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=true
    translation_scale=5.0
    ;;
  5)
    MODEL_NAME=a12_bb3
    SE3_ENABLED=true
    translation_scale=10.0
    ;;
  6)
    MODEL_NAME=a12_bb9
    SE3_ENABLED=false
    translation_scale=0.0
    ;;
  7)
    MODEL_NAME=a12_bb9
    SE3_ENABLED=true
    translation_scale=0.0
    ;;
esac

# Denoiser checkpoint
# MODEL_NAME=a12_bb3
AD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/ad-step550000-epoch3313.ckpt

# Wandb settings
WANDB_ID=richardshuai
GROUP=a12_se3

# Set the experiment name to include the current s_max_override value
EXP_NAME=${MODEL_NAME}_se3${SE3_ENABLED}_ts${translation_scale}

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
