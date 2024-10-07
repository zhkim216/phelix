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
#SBATCH --array=0-95%16

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
c1_values=(0.6 0.7 0.8)
c2_values=(1.0 1.1 1.2 1.4)
tc_values=(0.4 0.5)
churn_values=(20 50 100 200)

# Compute total number of combinations
num_c1=${#c1_values[@]}
num_c2=${#c2_values[@]}
num_tc=${#tc_values[@]}
num_churn=${#churn_values[@]}
total_jobs=$(( num_c1 * num_c2 * num_tc * num_churn ))  # 3 * 4 * 2 * 4 = 96

# Compute indices based on SLURM_ARRAY_TASK_ID
task_id=$SLURM_ARRAY_TASK_ID

c1_index=$(( task_id / (num_c2 * num_tc * num_churn) ))
remainder=$(( task_id % (num_c2 * num_tc * num_churn) ))

c2_index=$(( remainder / (num_tc * num_churn) ))
remainder=$(( remainder % (num_tc * num_churn) ))

tc_index=$(( remainder / num_churn ))
churn_index=$(( remainder % num_churn ))

# Assign parameter values
c1=${c1_values[$c1_index]}
c2=${c2_values[$c2_index]}
tc=${tc_values[$tc_index]}
churn=${churn_values[$churn_index]}

# Wandb settings
WANDB_ID=richardshuai
GROUP=${MODEL_NAME}_s2

# Set the experiment name to include the current parameter values
EXP_NAME=${MODEL_NAME}_c1${c1}_c2${c2}_tc${tc}_churn${churn}

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
    noise_schedule.name=step_scale_interval \
    +noise_schedule.c1=${c1} \
    +noise_schedule.c2=${c2} \
    +noise_schedule.t_c=${tc} \
    churn_cfg.s_churn=${churn} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
