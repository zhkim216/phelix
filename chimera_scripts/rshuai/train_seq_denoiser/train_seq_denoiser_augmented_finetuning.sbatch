#!/bin/bash
#SBATCH --job-name=aasd0_a2
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
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
EXP_NAME=aasd0_a2
GROUP=augmented_af3_monomer_v2

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v2
CLUSTER_SAMPLE=true
FINETUNE_CKPT_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.cluster_sample=true \
    data.subset_length_range=[32,256] \
    data.n_train_cluster_resample=10 \
    data.max_scrmsd=2.0 \
    data.max_rel_rog=1.4 \
    train.batch_size=32 \
    model.task=allatom_seq_des \
    finetuning.enabled=true \
    finetuning.ckpt_path=${FINETUNE_CKPT_PATH} \
    denoiser=fampnn_denoiser \
    denoiser.per_residue_eps=true \
    denoiser.augment_eps=0.5 \
    denoiser.fampnn.use_gvp=true \
    denoiser.fampnn.n_layers=5 \
    denoiser.confidence_module.enabled=true \
    optim.optimizer=adam_inv_sqrt \
    optim.adam_inv_sqrt.ref_lr=5.0e-5 \
    optim.adam_inv_sqrt.ref_steps=10000 \
    train.compile_model=true \
    checkpointing.save_latest_every_n_steps=5000 \
    trainer.check_val_every_n_epoch=3 \
    hydra.run.dir=${HYDRA_DIR}