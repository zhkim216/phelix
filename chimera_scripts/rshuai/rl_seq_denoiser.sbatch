#!/bin/bash
#SBATCH --job-name=d3_aasd0
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

# Run settings
EXP_NAME=d3_aasd0
GROUP=seq_denoiser_d3

# Dataset paths
CKPT_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt
TRAIN_PDB_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/ingraham_cath_L50_256_S50
VAL_PDB_PATH=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/ingraham_cath_L50_256_S50_val

# Run training script with the selected eps
WANDB__SERVICE_WAIT=300 python3 allatom_design/rl_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    no_wandb=false \
    ckpt_path=${CKPT_PATH} \
    train_pdb_path=${TRAIN_PDB_PATH} \
    val_pdb_path=${VAL_PDB_PATH} \
    train.batch_size=32 \
    train.compile_model=true \
    data.min_margin=3.0 \
    data.winner_threshold=3.0 \
    loss.loss_weights.alpha=0.0 \
    checkpointing.save_latest_every_n_steps=10000 \
    checkpointing.save_ema_every_n_steps=5000 \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}