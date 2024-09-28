#!/bin/bash
#SBATCH --job-name=a8_bb4_eval_bb_gen_scg_array
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-5

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Define directories and paths
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv
NNTM_DATASET=null  # disable for array jobs

# Model name
MODEL_NAME=a8_bb4
GROUP=chimera_a8_s2

# Directory where denoiser was trained
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Define parameter options
w_options=(0.5 0.75 1.0 1.2 1.5 2.0)

# Calculate indices based on SLURM_ARRAY_TASK_ID
w_index=$(( SLURM_ARRAY_TASK_ID % 5 ))     # 0 to 4 for 'w' values

# Assign parameter values
w=${w_options[$w_index]}
EXP_NAME=${MODEL_NAME}_eval_bb_gen_scg_w${w}

# Run the evaluation script with the current parameter combination
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_bb_gen_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_epoch=1000 \
    eval_every_n_ckpts=2 \
    batch_size=64 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    length_range=[256,384] \
    num_steps_list=[500] \
    length_step_size=4 \
    n_samples_per_length=4 \
    autoguidance_cfg.use_autoguidance=false \
    mpnn.overrides.batch_size=1 \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
