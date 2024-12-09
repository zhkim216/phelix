#!/bin/bash
#SBATCH --job-name=aasd_eval_inv_fold
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
# Array jobs over all models except the starred one (b15_aasd0)
#SBATCH --array=0-18%2

# Write batch script to logs
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
PDB_PATH=/scratch/hielab/talal/datasets/af3_pdb
DESIGNABILITY_CSV=null

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Models to evaluate (excluding b15_aasd0)
MODEL_NAMES=( \
"b12_aasd1" \
"b12_aasd2" \
"b12_aasd3" \
"b12_aasd4" \
"b12_aasd6" \
"b12_aasd7" \
"b15_aasd1" \
"b15_aasd2" \
"b15_aasd3" \
"b15_aasd4" \
"b15_aasd5" \
"b16_aasd0" \
"b16_aasd1" \
"b16_aasd2" \
"b16_aasd3" \
"b17_aasd0" \
"b17_aasd1" \
"b17_aasd2" \
"b17_aasd3" \
)

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

BASE_GROUP=${MODEL_NAME%%_aasd*}
GROUP="chimera_${BASE_GROUP}"

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_inv_fold

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
    num_steps_list=[20] \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    subset_length_range=[64,512] \
    hydra.run.dir=${HYDRA_DIR}
