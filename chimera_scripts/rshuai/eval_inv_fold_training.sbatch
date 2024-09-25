#!/bin/bash
#SBATCH --job-name=a7_aasd2_eval_inv_fold_codes32
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Model name to evaluate
MODEL_NAME=a7_aasd2

EXP_NAME=${MODEL_NAME}_eval_inv_fold_codes32
GROUP=chimera_a7

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_inv_fold_training.py \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    start_epoch=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    run_seq_recovery=false \
    run_codes_sc=true \
    num_codes_sc_samples=32 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    subset_length_range=[64,256] \
    hydra.run.dir=${HYDRA_DIR}