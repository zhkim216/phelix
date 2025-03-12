#!/bin/bash
#SBATCH --job-name=d0_rl_aasd0
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh

conda activate allatom_design

BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design
WANDB_ID=richardshuai
cd ${BASE_DIR}

# Dataset paths
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/b17_aasd9/post_hoc_ema_ckpts/ema-step200000-std0.220.ckpt
TRAIN_PDB_PATH=/scratch/users/rshuai/allatom_design/allatom_design/b17_aasd9/sample_multi_cath/ingraham_cath_L50_150_S50_T0.5
VAL_PDB_PATH=/scratch/users/rshuai/allatom_design/allatom_design/b17_aasd9/sample_multi_cath/ingraham_cath_L50_150_S50_T0.5_val

# Run settings
EXP_NAME=d0_rl_aasd0
GROUP=sherlock_d0_rl

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
    train.batch_size=16 \
    train.compile_model=false \
    checkpointing.save_latest_every_n_steps=20000 \
    checkpointing.save_ema_every_n_steps=5000 \
    trainer.check_val_every_n_epoch=5 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
