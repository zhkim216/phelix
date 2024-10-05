#!/bin/bash
#SBATCH --job-name=a11_bb2
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
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
EXP_NAME=a11_bb2
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
    train.batch_size=16 \
    train.compile_model=true \
    model.task=backbone \
    sd_interpolant.full_noise_p=0.3 \
    sd_interpolant.training_noise_schedule=uniform_t \
    denoiser=esm_dit_denoiser \
    denoiser.esm.esm_type=esm2_650M \
    denoiser.autoguidance.enabled=false \
    denoiser.training_batch_size_mult=8 \
    eval.eval_timesteps_seq=[0.25,0.5,0.75,1.0] \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=80 \
    checkpointing.save_ema_every_n_epochs=10 \
    trainer.check_val_every_n_epoch=40 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
