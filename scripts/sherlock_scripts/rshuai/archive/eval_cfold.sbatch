#!/bin/bash
#SBATCH --job-name=sd4_a5_cfold2
#SBATCH -p possu
#SBATCH --array=0-2
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
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Inputs config
CFOLD_CONFORMER_DIR=/scratch/users/rshuai/datasets/cfold/conformers
PDB_NAME_LIST=/scratch/users/rshuai/datasets/cfold/pdb_name_lists/conformer_dirs_L32_256.txt

CONFIGS=(
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a5/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a5/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt +sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=5000 +sampling_cfg_overrides.potts_sampling_cfg.regularization=LCP"
"ckpt_path=/scratch/users/rshuai/allatom_design/allatom_design/sd4_a5/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt +sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=5000 +sampling_cfg_overrides.potts_sampling_cfg.regularization=none"
)
OVERRIDES="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
BASE_EXP_NAME=sd4_a5_cfold2
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/eval_cfold/${BASE_EXP_NAME}

# Wandb settings / experiment name
WANDB_ID=richardshuai
GROUP=seq_des
EXP_NAME=${BASE_EXP_NAME}_idx${SLURM_ARRAY_TASK_ID}_${STRUCT_PRED_MODEL}

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/eval_cfold.py \
    wandb.no_wandb=true \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    input_cfg.conformer_dir=${CFOLD_CONFORMER_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    af2.data_dir=${AF2_DATA_DIR} \
    run_self_consistency_eval=false \
    run_af2rank=true \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=1 \
    ${OVERRIDES} \
    hydra.run.dir=${HYDRA_DIR}