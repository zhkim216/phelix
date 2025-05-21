#!/bin/bash
#SBATCH --job-name=LP_s5_a4_p1.0
#SBATCH -p possu
#SBATCH --nodelist=sh03-18n05
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.1
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software  # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.
# Sequence design model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
FAMPNN_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt
NOISE_LABELS=0.0

# Structure prediction model params
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Denoiser path
MODEL_NAME=s5_a4_p1.0
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Input config
PDB_DIR=/scratch/users/rshuai/datasets/augmented_af3_monomer_v2_boltz/preprocessing/process_pdbs/mmcifs
PDB_NAME_LIST=/scratch/users/rshuai/datasets/augmented_af3_monomer_v2_boltz/pdb_name_lists/eval_L32_256.txt
N_SUBSAMPLE=96

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_scaffold_training_${PHASE}
GROUP=backbone

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_scaffold_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    seq_des_cfg.fampnn.fampnn_ckpt=${FAMPNN_CKPT} \
    seq_des_cfg.fampnn.overrides.noise_labels=${NOISE_LABELS} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    input_cfg.pdb_name_ext=${PDB_NAME_EXT} \
    input_cfg.n_subsample=${N_SUBSAMPLE} \
    start_step=0 \
    eval_every_n_ckpts=4 \
    hydra.run.dir=${HYDRA_DIR}
