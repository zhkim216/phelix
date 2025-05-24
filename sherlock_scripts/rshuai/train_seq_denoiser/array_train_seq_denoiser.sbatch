#!/bin/bash
#SBATCH --job-name=sd3_a6
#SBATCH -p bioe
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=sh03-18n05
#SBATCH --time=96:00:00
#
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
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
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Wandb settings / experiment name
EXP_NAME_BASE=sd3_a6
GROUP=seq_des

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/boltz_v2

CONFIGS=(
"denoiser.mpnn.polymer_diffusion.t_max=1.0 denoiser.mpnn.polymer_diffusion.canonicalize=true"
"denoiser.mpnn.polymer_diffusion.t_max=0.75 denoiser.mpnn.polymer_diffusion.canonicalize=true"
"denoiser.mpnn.polymer_diffusion.t_max=0.5 denoiser.mpnn.polymer_diffusion.canonicalize=true"
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
    data.max_tokens=512 \
    data.max_atoms=4608 \
    train.batch_size=8 \
    trainer.accumulate_grad_batches=4 \
    train.compile_model=true \
    denoiser.per_residue_eps=false \
    denoiser.augment_eps=0.0 \
    denoiser.mpnn.polymer_diffusion.enabled=true \
    mask_selector.restype_masking_schedule=constant_t \
    denoiser.mpnn.use_atom_encoder=false \
    denoiser.mpnn.token_features.ca_only=false \
    data.filters.4.max_residues=2048 \
    ${OVERRIDES} \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR}