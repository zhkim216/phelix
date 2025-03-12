#!/bin/bash
#SBATCH --job-name=a12_bb9_ss_churn_sweep
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
#SBATCH --array=0-11

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

# Denoiser checkpoint
MODEL_NAME=a12_bb9
AD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/post_hoc_ema_ckpts/ema-step650000-std0.150.ckpt

# Grid search parameter values
step_scale_values=(1.0 1.1 1.2)
churn_values=(20 50 100 200)

# Compute indices based on SLURM_ARRAY_TASK_ID
task_id=$SLURM_ARRAY_TASK_ID
step_scale=${step_scale_values[$((task_id / 4))]}
churn=${churn_values[$((task_id % 4))]}

# Wandb settings
WANDB_ID=richardshuai
GROUP=${MODEL_NAME}_ss_churn

# Set the experiment name to include the current parameter values
EXP_NAME=${MODEL_NAME}_ss${step_scale}_churn${churn}

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/draw_bb_samples.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    ad_ckpt=${AD_CKPT} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    ss_kld.dssp_csv=${DSSP_CSV} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    n_samples_per_length=10 \
    noise_schedule.name=step_scale \
    +noise_schedule.c=${step_scale} \
    churn_cfg.s_churn=${churn} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
