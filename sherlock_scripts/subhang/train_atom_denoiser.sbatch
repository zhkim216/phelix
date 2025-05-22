#!/bin/bash
#SBATCH --job-name=ai_dit_12b_ema99
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/subhang/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/subhang/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/subhang/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.subhang_bash_profile
ml python
module load gcc/12.4.0
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories
BASE_DIR=/home/users/rshuai/subhang/allatom-design
HYDRA_DIR=/scratch/users/rshuai/subhang/hydra_outputs
OUT_DIR=/scratch/users/rshuai/subhang/allatom_design

cd ${BASE_DIR}

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/subhang/cache/torch

# Set experiment name and grouop
WANDB_ID=null
EXP_NAME=ai_dit_12b_ema99
GROUP=backbone

# Dataset paths
# PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset
PDB_PATH=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree
CLUSTER_SAMPLE=false

# PDB_PATH=/scratch/users/rshuai/datasets/af3_pdb_monomer
# CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=8 \
    no_wandb=false \
    data.pdb_paths=[${PDB_PATH}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.fixed_size=256 \
    data.subset_length_range=[32,256] \
    train.batch_size=64 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
