#!/bin/bash
#SBATCH --job-name=run_sc_eval
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

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
export SOFTWARE_PATH=/scratch/hielab/rshuai/software

# Model params
BOLTZ1_CHECKPOINT=/scratch/hielab/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Inputs config
PDB_DIR=/scratch/hielab/rshuai/datasets/boltz_val_cifs/pdbs
PDB_LIST_STEM=protein_monomer_32_256
PDB_NAME_LIST=/scratch/hielab/rshuai/datasets/boltz_val_cifs/pdb_name_lists/${PDB_LIST_STEM}.txt
LIGANDMPNN_OUTPUTS_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/boltz_val_cifs_evals/${PDB_LIST_STEM}/ligandmpnn/designs
MODEL_NAME=boltz1

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=ligandmpnn_bvc_${PDB_LIST_STEM}_${MODEL_NAME}_1sample
GROUP=seq_des
BASE_OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/boltz_val_cifs_evals/${PDB_LIST_STEM}/ligandmpnn

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/benchmarking/ligandmpnn_sc_eval.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CHECKPOINT} \
    struct_pred_cfg.model_name=${MODEL_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    ligandmpnn_outputs_dir=${LIGANDMPNN_OUTPUTS_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    max_samples_per_pdb=1 \
    hydra.run.dir=${HYDRA_DIR}
