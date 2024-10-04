#!/bin/bash
#SBATCH --job-name=a10_bb3_draw_bb_samples
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
# Array job
#SBATCH --array=0-24%10

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
NNTM_DATASET=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/pdb_store_renumbered
DSSP_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/dssp/dssp_length_strict.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Denoiser checkpoint
MODEL_NAME=a10_bb3
AD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/ad-epoch1279.ckpt

# Define modes for array tasks
MODES=("linear" "square" "cubic" "sqrt" "cbrt")
ca_index=$(( SLURM_ARRAY_TASK_ID / 5 ))
nco_index=$(( SLURM_ARRAY_TASK_ID % 5 ))
ca_mode=${MODES[$ca_index]}
nco_mode=${MODES[$nco_index]}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_draw_bb_samples_ca_${ca_mode}_nco_${nco_mode}
GROUP=chimera_${MODEL_NAME}_s1

# Navigate to the base directory
cd ${BASE_DIR}

# Run the evaluation script with appropriate parameters
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
    ca.timestep_schedule.mode=${ca_mode} \
    nco.timestep_schedule.mode=${nco_mode} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
