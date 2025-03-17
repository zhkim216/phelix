#!/bin/bash
#SBATCH --job-name=eval_seq_des
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

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Inputs config
PDB_DIR=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/eval_mmcifs
PDB_KEY_LIST=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/eval_pdb_keys.list

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Model to evaluate
MODEL_NAME=c6_aasd1
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}
NOISE_LABELS=0.1

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_seq_des_nl${NOISE_LABELS}_an
GROUP=chimera_c6

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_seq_des_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.model_name=esmfold \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_key_list=${PDB_KEY_LIST} \
    input_cfg.pdb_key_ext=".cif" \
    input_cfg.subset_length_range=[32,256] \
    input_cfg.n_subsample=256 \
    start_step=100000 \
    end_step=150000 \
    eval_every_n_ckpts=1 \
    sampling_cfg_overrides.add_noise=true \
    hydra.run.dir=${HYDRA_DIR}
