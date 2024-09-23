#!/bin/bash
#SBATCH --job-name=a5_cbb14_sampling_sweep
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-143%30                # 144 tasks with max 30 concurrent jobs
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#

## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Define directories
BASE_DIR=/home/talal/rshuai/protein-ldm
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv
NNTM_DATASET=null  # don't run nntm since it breaks with large sweeps

# Model name to evaluate
MODEL_NAME=a5_cbb14

GROUP=chimera_a5_s0

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Define parameter grids
ca_modes=("linear" "square" "cubic" "sqrt")
nco_modes=("linear" "square" "cubic" "sqrt")
noise_cs=(1.0 1.2 1.4)

# Total combinations: 4 (ca.mode) * 4 (nco.mode) * 3 (ca_c) * 3 (nco_c) = 144
# Map SLURM_ARRAY_TASK_ID (0-143) to parameter indices

TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices
ca_mode_index=$(( TASK_ID / 36 ))                    # 0-3
nco_mode_index=$(( (TASK_ID / 9) % 4 ))              # 0-3
ca_c_index=$(( (TASK_ID / 3) % 3 ))             # 0-2
nco_c_index=$(( TASK_ID % 3 ))                  # 0-2

# Assign parameters based on indices
ca_mode=${ca_modes[$ca_mode_index]}
nco_mode=${nco_modes[$nco_mode_index]}
ca_c=${noise_cs[$ca_c_index]}
nco_c=${noise_cs[$nco_c_index]}

# Set a unique experiment name incorporating parameter values
EXP_NAME=${MODEL_NAME}_eval_sweep_ca${ca_mode}C${ca_c}_nco${nco_mode}C${nco_c}

# Run evaluation script with the selected parameters
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_bb_gen_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_epoch=2350 \
    eval_every_n_ckpts=2 \
    batch_size=64 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    length_range=[64,128] \
    num_steps_list=[500] \
    length_step_size=2 \
    n_samples_per_length=2 \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR} \
    timestep_schedule.ca.mode=${ca_mode} \
    ca_diffusion.noise_schedule.name=step_scale \
    +ca_diffusion.noise_schedule.c=${ca_c} \
    timestep_schedule.nco.mode=${nco_mode} \
    nco_diffusion.noise_schedule.name=step_scale \
    +nco_diffusion.noise_schedule.c=${nco_c}
