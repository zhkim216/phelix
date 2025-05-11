#!/bin/bash
#SBATCH --job-name=sd2_a0_res
#SBATCH -p bioe
#SBATCH --array=0-2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Wandb settings / experiment name
EXP_NAME_BASE=sd2_a0
GROUP=seq_des

# Dataset paths
PDB_PATH=/scratch/users/subhang/datasets/boltz_v2

CONFIGS=(
"data.filters.2.resolution=2.0"
"data.filters.2.resolution=3.5"
"data.filters.2.resolution=5.0"
)
OVERRIDES="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAME_BASE}_cfg${SLURM_ARRAY_TASK_ID}"

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_path=${PDB_PATH} \
    data.max_tokens=256 \
    data.max_atoms=4608 \
    data.filters.1=null \
    data.filters.4.max_residues=512 \
    data.filters.4.min_residues=32 \
    train.batch_size=16 \
    trainer.accumulate_grad_batches=2 \
    train.compile_model=false \
    denoiser.per_residue_eps=false \
    denoiser.augment_eps=0.3 \
    denoiser.mpnn.k_neighbors=48 \
    denoiser.mpnn.use_atom_encoder=false \
    denoiser.mpnn.atom_encoder.n_layers=1 \
    denoiser.mpnn.atom_encoder.k_atom_neighbors=16 \
    denoiser.mpnn.atom_encoder.atom_s=64 \
    denoiser.mpnn.atom_encoder.atom_z=16 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    ${OVERRIDES} \
    hydra.run.dir=${HYDRA_DIR}