#!/bin/bash
#SBATCH --job-name=sd4_a1_ens1
#SBATCH -p bioe
#SBATCH --array=0-5
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
LIST_STEM=protein_monomer_32_512_no_ligand
# CONFORMER_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/conformers/${LIST_STEM}/md_pd_conformers-sde-aicath-no-recurrence/sde-aicath-no-recurrence-richards-structures-90
BASE_CONFORMER_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/conformers/${LIST_STEM}/md_pd_conformers-sde-aicath-no-recurrence/sde-aicath-no-recurrence-richards-structures-

CONFIGS=(
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}70"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}80"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}90"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}100"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}110"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a1_cfg0/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt conformer_dir=${BASE_CONFORMER_DIR}120"
)
OVERRIDES="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
BASE_EXP_NAME=sd4_a1_ens1
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/ensemble_design/${BASE_EXP_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
GROUP=seq_des
EXP_NAME=${BASE_EXP_NAME}_idx${SLURM_ARRAY_TASK_ID}_bvc_${LIST_STEM}_${STRUCT_PRED_MODEL}

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/seq_des_multi_ensemble.py \
    wandb.no_wandb=false \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    run_self_consistency_eval=true \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=8 \
    ${OVERRIDES} \
    hydra.run.dir=${HYDRA_DIR}