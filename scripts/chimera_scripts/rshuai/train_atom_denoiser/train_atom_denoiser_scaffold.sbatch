#!/bin/bash
#SBATCH --job-name=s5_a1_mmdit
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
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=s5_a1_mmdit
GROUP=backbone

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v2_boltz

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_path=${PDB_PATH} \
    data.filters.3.max_scrmsd=1.0 \
    task=scaffold \
    motif_selector.atom_p.all=0.0 \
    motif_selector.atom_p.protein_sidechain=0.0 \
    motif_selector.atom_p.protein_backbone=1.0 \
    motif_selector.residx_mask_p.all=0.0 \
    motif_selector.residx_mask_p.none=1.0 \
    motif_selector.residx_mask_p.uniform=0.0 \
    denoiser=dit_denoiser \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    denoiser.dit.num_heads=12 \
    denoiser.dit.use_mmdit=true \
    train.batch_size=16 \
    denoiser.training_batch_size_mult=8 \
    data.samples_per_epoch=16000 \
    train.compile_model=true \
    hydra.run.dir=${HYDRA_DIR}
