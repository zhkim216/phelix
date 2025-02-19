#!/bin/bash
#SBATCH --job-name=eval_sc
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
GROUP=eval_sc_dpo
MODEL_NAME=d2_aasd0
CKPT=sd-step100000-epoch36.ckpt
# MODEL_NAME=c2_aasd2
EXP_NAME=${MODEL_NAME}_eval_sc_dpo_nl0.3

# Set paths
MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/${CKPT}
# MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt
PDB_DIR=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/pdb_store
PDB_KEY_LIST=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/eval_pdb_keys.list

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_sc_dpo.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    subset_length_range=[50,256] \
    noise_labels=0.3 \
    num_pdbs=300
