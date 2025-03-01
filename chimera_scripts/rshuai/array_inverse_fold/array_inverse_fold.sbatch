#!/bin/bash
#SBATCH --job-name=b8_aasd6_inv_fold
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-11%6
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and activate conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model parameters
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Sweep parameters
num_steps_values=(1 10 20 100)
aatype_decoding_order_modes=(random greedy greedy_psce)

# Calculate indices
num_num_steps=${#num_steps_values[@]}                    # 4
num_aatype_modes=${#aatype_decoding_order_modes[@]}      # 3

total_jobs=$(( num_num_steps * num_aatype_modes ))        # 12

# Ensure the SLURM_ARRAY_TASK_ID is within the range
if (( SLURM_ARRAY_TASK_ID >= total_jobs )); then
  echo "Error: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) exceeds total_jobs (${total_jobs})"
  exit 1
fi

# Determine indices for num_steps and aatype_decoding_order_mode
num_steps_idx=$(( SLURM_ARRAY_TASK_ID / num_aatype_modes ))          # 0-3
aatype_decoding_order_mode_idx=$(( SLURM_ARRAY_TASK_ID % num_aatype_modes ))  # 0-2

# Assign variables based on indices
NUM_STEPS=${num_steps_values[$num_steps_idx]}
AATYPE_DECODING_ORDER_MODE=${aatype_decoding_order_modes[$aatype_decoding_order_mode_idx]}

# Fixed model and checkpoint
MODEL_NAME="b8_aasd6"
CKPT="sd-step100000-epoch300.ckpt"

# Denoiser checkpoint path
SD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/${CKPT}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_sd_${CKPT}_S${NUM_STEPS}_${AATYPE_DECODING_ORDER_MODE}
GROUP=chimera_b8_aasd6_inv_fold

# Run inverse folding script
cd ${BASE_DIR} || { echo "Base directory not found: ${BASE_DIR}"; exit 1; }

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/inverse_fold.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    sd_ckpt=${SD_CKPT} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    data.pdb_path=${PDB_PATH} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    temperature=0.0 \
    timestep_schedule.num_steps=${NUM_STEPS} \
    aatype_decoding_order_mode=${AATYPE_DECODING_ORDER_MODE} \
    num_pdbs=300 \
    num_seqs_per_pdb=1 \
    batch_size=64 \
    hydra.run.dir=${HYDRA_DIR}
