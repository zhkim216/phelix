#!/bin/bash
#SBATCH --job-name=u4_a1_t0_0.5
#SBATCH -p possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=sh04-04n01
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Set experiment name and grouop
WANDB_ID=richardshuai
EXP_NAME=u4_a1_t0_0.5
GROUP=backbone

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/augmented_af3_monomer_v2
CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_paths=[${PDB_PATH}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.max_scrmsd=1.0 \
    denoiser=dit_denoiser \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    +denoiser.interpolant.training_noise_cfg.lognormal.t_boundaries=[0.0,0.5] \
    train.batch_size=128 \
    train.compile_model=true \
    hydra.run.dir=${HYDRA_DIR}
