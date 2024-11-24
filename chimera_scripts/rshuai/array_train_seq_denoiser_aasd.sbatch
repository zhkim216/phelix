#!/bin/bash
#SBATCH --job-name=b8_aasd4_noise_schedule
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-4%5
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

# Define arrays for psigma_mean and psigma_std
psigma_mean_values=(1.0 1.0 1.5 2.0 2.0)
psigma_std_values=(1.5 1.0 1.2 1.0 1.2)

# Calculate indices
total_jobs=${#psigma_mean_values[@]}

psigma_idx=$SLURM_ARRAY_TASK_ID

# Get the current psigma_mean and psigma_std based on index
PSIGMA_MEAN=${psigma_mean_values[$psigma_idx]}
PSIGMA_STD=${psigma_std_values[$psigma_idx]}

# Set experiment name and group, including psigma_mean and psigma_std
EXP_NAME=b8_aasd4_mean${PSIGMA_MEAN}_std${PSIGMA_STD}
GROUP=chimera_b8

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script with the selected psigma_mean and psigma_std
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.subset_length_range=[16,9999] \
    train.batch_size=64 \
    train.compile_model=true \
    model.task=allatom_seq_des \
    interpolant.training_noise_schedule=uniform_sqrt_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.model_type=sidechain \
    denoiser.minimpnn.augment_eps=0.3 \
    denoiser.scn_diffusion_module.training_batch_size_mult=8 \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=5000 \
    checkpointing.save_ema_every_n_steps=1500 \
    trainer.check_val_every_n_epoch=10 \
    denoiser.interpolant.training_noise_cfg.lognormal.psigma_mean=${PSIGMA_MEAN} \
    denoiser.interpolant.training_noise_cfg.lognormal.psigma_std=${PSIGMA_STD} \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
