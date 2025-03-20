#!/bin/bash
#SBATCH --job-name=u1_a7_refactor_test
#SBATCH -p gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
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
EXP_NAME=u1_a7_refactor_test
GROUP=augmented_af3_monomer_v1

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/augmented_af3_monomer_v1
CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    num_workers=8 \
    data.pdb_paths=[${PDB_PATH}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.fixed_size=256 \
    data.subset_length_range=[32,256] \
    data.max_scrmsd=2.0 \
    data.max_rel_rog=1.4 \
    data.n_train_cluster_resample=10 \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    train.batch_size=128 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    trainer.check_val_every_n_epoch=4 \
    hydra.run.dir=${HYDRA_DIR}