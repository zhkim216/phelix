#!/bin/bash
#SBATCH --job-name=seq_des_multi_multistate
#SBATCH -p bioe
#SBATCH --array=0-11
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
EPS_VALUES=(0.1 0.3 0.5 1.0)
MAX_NUM_CONFORMERS_VALUES=(2 8 16)
BATCH_SIZE_VALUES=(8 2 1)  # 8, 2, 1 for 2, 8, 16 conformers to keep batch size constant at 16
TASK_ID=${SLURM_ARRAY_TASK_ID}
EPS_IDX=$(( TASK_ID % ${#EPS_VALUES[@]} ))
CONF_IDX=$(( TASK_ID / ${#EPS_VALUES[@]} ))
EPS=${EPS_VALUES[$EPS_IDX]}
MAX_NUM_CONFORMERS=${MAX_NUM_CONFORMERS_VALUES[$CONF_IDX]}
BATCH_SIZE=${BATCH_SIZE_VALUES[$CONF_IDX]}
LIST_STEM=protein_monomer_32_512_no_ligand
CONFORMER_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/conformers/${LIST_STEM}/gaussian_conformers/eps${EPS}

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a2_cfg1/checkpoints/ema/sd-step17500-epoch22-ema0.99.ckpt
BASE_OUT_DIR=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a2_cfg1/multistate_sweep

# Wandb settings / experiment name
WANDB_ID=richardshuai
GROUP=seq_des
EXP_NAME=sd3_a2_cfg1_ms_sweep1_eps${EPS}_nc${MAX_NUM_CONFORMERS}_bvc_${LIST_STEM}_${STRUCT_PRED_MODEL}

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/seq_des_multi_multistate.py \
    wandb.no_wandb=false \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    ckpt_path=${CKPT_PATH} \
    base_out_dir=${BASE_OUT_DIR} \
    conformer_dir=${CONFORMER_DIR} \
    max_num_conformers=${MAX_NUM_CONFORMERS} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    run_self_consistency_eval=true \
    sampling_cfg_overrides.batch_size=${BATCH_SIZE} \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    +sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=5000 \
    +sampling_cfg_overrides.potts_sampling_cfg.regularization=null \
    hydra.run.dir=${HYDRA_DIR}