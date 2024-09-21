#!/bin/bash
#SBATCH --job-name=a5_cbb_eval_array
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-24
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#

## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_protein_ldm

# Define directories
BASE_DIR=/home/talal/rshuai/protein-ldm
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv
NNTM_DATASET=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/pdb_store_renumbered

# Set MODEL_NAME based on $SLURM_ARRAY_TASK_ID
MODEL_NAME=a5_cbb${SLURM_ARRAY_TASK_ID}

# Set EXP_NAME based on MODEL_NAME
EXP_NAME=${MODEL_NAME}_eval_bb_gen
GROUP=chimera_a5

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/protein_ldm/allatom_design/${MODEL_NAME}

# Run evaluation script with the selected parameters
WANDB__SERVICE_WAIT=300 python3 protein_ldm/eval/eval_bb_gen.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_epoch=0 \
    eval_every_n_ckpts=2 \
    batch_size=64 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=${NNTM_DATASET} \
    length_range=[64,128] \
    num_steps_list=[500] \
    length_step_size=4 \
    n_samples_per_length=2 \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
