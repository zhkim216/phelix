#!/bin/bash
#SBATCH --job-name=a11_bb1
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
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
cd $BASE_DIR

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=a11_bb1
GROUP=chimera_a11

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.fixed_size=512 \
    train.batch_size=64 \
    train.compile_model=true \
    model.task=backbone \
    denoiser=dit_denoiser \
    denoiser.autoguidance.enabled=true \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=40 \
    checkpointing.save_ema_every_n_epochs=10 \
    trainer.check_val_every_n_epoch=20 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
