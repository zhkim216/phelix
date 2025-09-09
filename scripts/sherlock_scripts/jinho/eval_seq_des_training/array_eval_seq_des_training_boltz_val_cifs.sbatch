#!/bin/bash
#SBATCH --job-name=tk_bds_rss
#SBATCH -p bioe,possu
#SBATCH --array=0-3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_eval_seq_des_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_eval_seq_des_%A_%a.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

source ~/.bashrc

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.4
venv enzyme_design

# Define directories and paths
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir
SOFTWARE_PATH=/oak/stanford/groups/possu/jinho/software  # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.

# Model params
STRUCT_PRED_MODEL=boltz1x
BOLTZ1_CKPT=/scratch/users/zhkim216/model_params/boltz1/boltz1x_conf.ckpt

# Inputs config
PDB_DIR=/scratch/users/zhkim216/datasets/boltz_val_cifs/pdbs
LIST_STEM=protein_monomer_32_256
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/boltz_val_cifs/pdb_name_lists/${LIST_STEM}.txt

# Set cache directories
export HF_HOME=/scratch/users/zhkim216/cache/huggingface
export TORCH_HOME=/scratch/users/zhkim216/cache/torch

# Model to evaluate
MODEL_NAMES=(
    "token_bonds_cfg2"
    "token_bonds_cfg3"
    "token_bonds_cfg4"
    "token_bonds_cfg5"
)
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}
MODEL_CFG_NAME="ligand_potts"

# Sampling configs to use
SAMPLING_CFG_PATH="/home/users/zhkim216/code/allatom-design/allatom_design/configs/seq_des/ligand_potts_inference.yaml"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT="ligand_seq_des"
GROUP=eval
EXP_NAME=${MODEL_NAME}_eval_seq_des_bvc_${LIST_STEM}_${STRUCT_PRED_MODEL}

# REDIS PORT (Avoid using the same port with Richard)
REDIS_PORT=1296

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_seq_des_training.py \
    --config-dir allatom_design/configs/eval \
    --config-name eval_seq_des_training.yaml \
    wandb.wandb_id=${WANDB_ID} \
    wandb.project=${PROJECT} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.model_name=${MODEL_CFG_NAME} \
    seq_des_cfg.atom_mpnn.sampling_cfg=${SAMPLING_CFG_PATH} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_processing_cfg.ccd_cfg.redis_port=${REDIS_PORT} \
    start_step=0 \
    end_step=null \
    eval_every_n_ckpts=1 \
    hydra.run.dir=${HYDRA_DIR}
