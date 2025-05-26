#!/bin/bash
#SBATCH --job-name=t0.5_bvc_eval_seq_des
#SBATCH -p bioe,possu
#SBATCH --array=0-7
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software  # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.

# Model params
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Inputs config
PDB_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/pdbs
LIST_STEM=protein_monomer_32_256
PDB_NAME_LIST=/scratch/users/rshuai/datasets/boltz_val_cifs/pdb_name_lists/${LIST_STEM}.txt

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to evaluate
MODEL_NAMES=(
    "sd3_a4"
    "sd3_a5_cfg0"
    "sd3_a5_cfg1"
    "sd3_a6_cfg0"
    "sd3_a6_cfg1"
    "sd3_a7_cfg0"
    "sd3_a7_cfg1"
    "sd3_a7_cfg2"
)
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_t0.5_eval_seq_des_bvc_${LIST_STEM}_${STRUCT_PRED_MODEL}
GROUP=seq_des

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_seq_des_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    start_step=0 \
    end_step=40000 \
    eval_every_n_ckpts=1 \
    +sampling_cfg_overrides.t=0.5 \
    hydra.run.dir=${HYDRA_DIR}
