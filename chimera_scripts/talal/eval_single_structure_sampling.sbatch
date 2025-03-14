#!/bin/bash
#SBATCH --job-name=eval_interface_redesign
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/talal/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/talal/slurm_log_files/job_%j.err
#

# Set base directories
BASE_DIR=/home/talal/allatom-design
HYDRA_DIR=/scratch/hielab/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/talal/cache/huggingface
export TORCH_HOME=/scratch/hielab/talal/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/talal/cache/omegafold/omegafold.pt

# Model to evaluate
CHECKPOINT_PATH=/scratch/hielab/talal/allatom_design/allatom_design/aasd_gvp_pdb_0.3_single_pack/checkpoints/sd-epoch53-step365040-val_loss1.2307.ckpt

# PDB to sample from
PDB_PATH=/home/talal/protein-dpo/data/af_pdbs/6cfg.pdb

# Wandb settings / experiment name
WANDB_ID=talal-widatalla
EXP_NAME=${MODEL_NAME}_eval_sample
GROUP=chimera_a7

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_single_structure_sampling.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    no_wandb=true \
    checkpoint_path=${CHECKPOINT_PATH}\
    num_workers=8 \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    aatype_decoding_order_mode=greedy \
    temperature=1.0 \
    fixed_pos=[0,1,2,3,4,5,6,7,8,9,10] \
    num_steps=5 \
    batch_size=2 \
    num_samples=2 \
    pdb_path=${PDB_PATH} \
    hydra.run.dir=${HYDRA_DIR} \
    struct_pred_cfg.model_name=esmfold \