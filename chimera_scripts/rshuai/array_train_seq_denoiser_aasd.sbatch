#!/bin/bash
#SBATCH --job-name=c1_aasd0_eps
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
WANDB_ID=richardshuai
cd $BASE_DIR

# Define arrays for eps sweep
EPS_ARRAY=(0.0 0.1 0.2 0.3)
EPS=${EPS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Set experiment name and group, including eps
EXP_NAME=c1_aasd0_eps${EPS}
GROUP=chimera_c1

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
# PDB_PATH=/scratch/hielab/talal/datasets/af3_pdb

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.subset_length_range=[16,9999] \
    train.batch_size=64 \
    train.compile_model=true \
    model.task=allatom_seq_des \
    interpolant.training_noise_schedule=uniform_sqrt_t \
    denoiser=fampnn_denoiser \
    denoiser.fampnn.augment_eps=${EPS} \
    denoiser.fampnn.per_residue_eps=true \
    denoiser.fampnn.use_gvp=true \
    denoiser.fampnn.n_layers=3 \
    denoiser.confidence_module.enabled=true \
    denoiser.confidence_module.structure_encoder.n_layers=3 \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=5000 \
    checkpointing.save_ema_every_n_steps=1500 \
    trainer.check_val_every_n_epoch=10 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}