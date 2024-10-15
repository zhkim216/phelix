#!/bin/bash
#SBATCH --job-name=a13_bb4
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
cd $BASE_DIR

# Set cache directories
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=a13_bb4
GROUP=chimera_a13

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
AFDB_PDB_PATH=/scratch/hielab/rshuai/datasets/afdb
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv
AFDB_DESIGNABILITY_CSV=null

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_paths=[${PDB_PATH},${AFDB_PDB_PATH}] \
    data.designability_csvs=[${DESIGNABILITY_CSV},null] \
    data.dataset_weights=[0.25,0.75] \
    data.fixed_size=512 \
    data.translation_scale=0.0 \
    train.batch_size=128 \
    train.compile_model=true \
    model.task=backbone \
    denoiser=dit_denoiser \
    denoiser.autoguidance.enabled=true \
    denoiser.dit.hidden_size=768 \
    denoiser.dit.num_heads=12 \
    denoiser.dit.translation_invariant=true \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=50000 \
    checkpointing.save_ema_every_n_steps=10000 \
    trainer.check_val_every_n_epoch=30 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
