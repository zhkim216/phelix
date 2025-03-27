#!/bin/bash
#SBATCH --job-name=c6_aasd1_eval_scn_pack
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
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
WANDB_ID=richardshuai

cd ${BASE_DIR}

# Inputs config
PDB_DIR=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/eval_mmcifs
PDB_NAME_LIST=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/eval_pdb_keys.list

# Model name to evaluate
MODEL_NAME=c6_aasd1

EXP_NAME=${MODEL_NAME}_eval_scn_pack
GROUP=seq_des

# Set HF transformers cache
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_scn_pack_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    input_cfg.pdb_name_ext=".cif" \
    input_cfg.subset_length_range=[64,512] \
    start_step=0 \
    eval_every_n_ckpts=1 \
    hydra.run.dir=${HYDRA_DIR}
