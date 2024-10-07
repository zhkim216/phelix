#!/bin/bash
#SBATCH --job-name=a11_bb1_draw_bb_samples
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
#SBATCH --array=0-104

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
MODEL_NAME=a11_bb1
AD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/post_hoc_ema_ckpts/ema-step1000000-std0.240.ckpt

# Grid search parameter values
step_scale_values=(0.7 0.8 0.9 1.0 1.1 1.2 1.4)
churn_values=(0 20 50 100 200)
w_values=(1.0 1.1 1.2)

# Compute total number of combinations
num_c=${#step_scale_values[@]}
num_churn=${#churn_values[@]}
num_w=${#w_values[@]}
total_jobs=$(( num_c * num_churn * num_w ))


c_index=$(( SLURM_ARRAY_TASK_ID / (num_churn * num_w) ))
remainder=$(( SLURM_ARRAY_TASK_ID % (num_churn * num_w) ))
churn_index=$(( remainder / num_w ))
w_index=$(( remainder % num_w ))

# Assign parameter values
c=${step_scale_values[$c_index]}
churn=${churn_values[$churn_index]}
w=${w_values[$w_index]}

# Wandb settings
WANDB_ID=richardshuai
GROUP=${MODEL_NAME}_autoguidance

# Set the experiment name to include the current autoguidance values
EXP_NAME=${MODEL_NAME}_ss${c}_churn${churn}_w${w}

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
    autoguidance_cfg.use_autoguidance=true \
    autoguidance_cfg.w=${w} \
    noise_schedule.name=step_scale \
    +noise_schedule.c=${c} \
    churn_cfg.s_churn=${churn} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
