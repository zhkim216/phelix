#!/bin/bash
#SBATCH --job-name=chroma_sc_eval_scaffolded_samples
#SBATCH -p bioe,possu
#SBATCH --array=0-447%15
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

# Model params (if running self-consistency eval)
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Inputs config
LANM_DESIGN_DIR=chroma_lanm_scaffolded
CHROMA_CONFIG=chroma_lanm_sweep1_manifest_idx${SLURM_ARRAY_TASK_ID}
INPUT_PDB_DIR=/scratch/users/rshuai/allatom_design/allatom_design/lanm_designs/${LANM_DESIGN_DIR}/${CHROMA_CONFIG}/designs/main_conformer
CHROMA_OUTPUTS_DIR=/scratch/users/rshuai/allatom_design/allatom_design/lanm_designs/${LANM_DESIGN_DIR}/${CHROMA_CONFIG}/designs
MODEL_NAME=boltz1

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${CHROMA_CONFIG}_boltz1
GROUP=lanm
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/lanm_designs/${LANM_DESIGN_DIR}/${CHROMA_CONFIG}

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/benchmarking/chroma_sc_eval.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    base_out_dir=${BASE_OUT_DIR} \
    chroma_outputs_dir=${CHROMA_OUTPUTS_DIR} \
    input_cfg.pdb_dir=${INPUT_PDB_DIR} \
    hydra.run.dir=${HYDRA_DIR}
