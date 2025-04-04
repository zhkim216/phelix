#!/bin/bash
#SBATCH --job-name=ablate_pretrained
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
OUT_DIR=/scratch/hielab/rshuai/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=s4_a7
GROUP=backbone

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v2
CLUSTER_SAMPLE=true

# Pretrained weights
PRETRAINED_FAMPNN_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_paths=[${PDB_PATH}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.max_scrmsd=1.0 \
    model.task=scaffold \
    trainer.strategy=ddp_find_unused_parameters_true \
    scaffold_manager=scaffold_manager \
    scaffold_manager.sidechain_dropout_p=0.5 \
    denoiser=dit_denoiser \
    denoiser.scaffold_module.enabled=true \
    pretrained_module_paths.fampnn=${PRETRAINED_FAMPNN_PATH} \
    denoiser.scaffold_module.ablate_pretrained_weights=true \
    denoiser.scaffold_module.freeze=true \
    denoiser.dit.scaffolding.use_concat=true \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    train.batch_size=16 \
    denoiser.training_batch_size_mult=8 \
    data.n_train_cluster_resample=2 \
    train.compile_model=true \
    hydra.run.dir=${HYDRA_DIR}
