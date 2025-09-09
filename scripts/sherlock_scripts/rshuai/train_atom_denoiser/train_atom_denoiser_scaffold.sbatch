#!/bin/bash
#SBATCH --job-name=s7_a0
#SBATCH -p possu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=sh03-18n05
#SBATCH --time=168:00:00
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
module load cuda/12.4
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
EXP_NAME=s7_a0
GROUP=backbone

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/boltz_v2

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_path=${PDB_PATH} \
    data.max_tokens=512 \
    data.cropper._target_=allatom_design.data.crop.boltz.BoltzCropper \
    data.use_auth_as_residx=false \
    task=scaffold \
    motif_selector.atom_p.all=0.0 \
    motif_selector.atom_p.protein_sidechain=0.0 \
    motif_selector.atom_p.protein_backbone=1.0 \
    motif_selector.residx_mask_p.all=0.0 \
    motif_selector.residx_mask_p.none=1.0 \
    motif_selector.residx_mask_p.uniform=0.0 \
    denoiser=dit_denoiser \
    denoiser.dit.pos_encoding=af3 \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    denoiser.dit.num_heads=12 \
    train.batch_size=4 \
    denoiser.training_batch_size_mult=8 \
    data.samples_per_epoch=8000 \
    train.compile_model=true \
    hydra.run.dir=${HYDRA_DIR}