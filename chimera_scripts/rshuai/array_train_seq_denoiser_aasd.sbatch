#!/bin/bash
#SBATCH --job-name=b4_aasd2_lr
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-5%3
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

# Define arrays for ref_lr and ref_steps
ref_lr_values=(5.0e-3 1.0e-3)
ref_steps_values=(10000 50000 100000)

# Calculate indices
num_ref_lr=${#ref_lr_values[@]}        # Should be 2
num_ref_steps=${#ref_steps_values[@]}  # Should be 3
total_jobs=$((num_ref_lr * num_ref_steps))  # Should be 6

ref_lr_idx=$(( SLURM_ARRAY_TASK_ID / num_ref_steps ))
ref_steps_idx=$(( SLURM_ARRAY_TASK_ID % num_ref_steps ))

# Get the current ref_lr and ref_steps based on indices
REF_LR=${ref_lr_values[$ref_lr_idx]}
REF_STEPS=${ref_steps_values[$ref_steps_idx]}

# Set experiment name and group, including ref_lr and ref_steps
EXP_NAME=b4_aasd2_lr${REF_LR}_steps${REF_STEPS}
GROUP=chimera_b4

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script with the selected ref_lr and ref_steps
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
    optim.optimizer=adam_inv_sqrt \
    optim.adam_inv_sqrt.ref_lr=${REF_LR} \
    optim.adam_inv_sqrt.ref_steps=${REF_STEPS} \
    checkpointing.save_latest_every_n_steps=5000 \
    checkpointing.save_ema_every_n_steps=1500 \
    trainer.check_val_every_n_epoch=10 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
