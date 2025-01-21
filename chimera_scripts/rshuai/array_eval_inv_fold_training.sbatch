#!/bin/bash
#SBATCH --job-name=if_test
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-5%2
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

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
# PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
# DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv
PDB_PATH=/scratch/hielab/talal/datasets/af3_pdb
DESIGNABILITY_CSV=null

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Arrays for model names and noise labels
model_names=(c2_aasd0 c2_aasd1)
noise_labels=(0.0 0.1 0.3)

# Determine which combination to run
model_idx=$((SLURM_ARRAY_TASK_ID / 3))
noise_idx=$((SLURM_ARRAY_TASK_ID % 3))

MODEL_NAME=${model_names[$model_idx]}
NOISE_LABEL=${noise_labels[$noise_idx]}

# Model to evaluate
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_inv_fold_nl${NOISE_LABEL}
GROUP=chimera_c2

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_inv_fold_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.model_name=esmfold \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    run_seq_recovery=false \
    run_codes_sc=true \
    num_codes_sc_batches=1 \
    temperature=0.1 \
    noise_labels=${NOISE_LABEL} \
    num_steps_list=[20] \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    subset_length_range=[64,512] \
    hydra.run.dir=${HYDRA_DIR}
