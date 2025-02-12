#!/bin/bash
#SBATCH --job-name=c3_aasd9
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
WANDB_ID=richardshuai
cd $BASE_DIR

EXP_NAME=c3_aasd9
GROUP=seq_denoiser_c3

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset

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
    denoiser=fampnn_denoiser \
    denoiser.augment_eps=0.3 \
    denoiser.per_residue_eps=true \
    denoiser.fampnn.use_gvp=false \
    denoiser.fampnn.n_layers=3 \
    denoiser.esm.use_esmc=true \
    denoiser.fampnn.init_hV_with_hS=true \
    trainer.check_val_every_n_epoch=10 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
