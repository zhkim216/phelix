#!/bin/bash
#SBATCH --job-name=s4_a0
#SBATCH -p possu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=sh03-18n05
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
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Set experiment name and grouop
WANDB_ID=richardshuai
EXP_NAME=s4_a0
GROUP=backbone

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/augmented_af3_monomer_v2
CLUSTER_SAMPLE=true

RESUME_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/s4_a0/checkpoints/ad-epoch1399.ckpt

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    resume.ckpt_path=${RESUME_CKPT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_paths=[${PDB_PATH}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    model.task=scaffold \
    scaffold_manager=scaffold_manager \
    denoiser=dit_denoiser \
    denoiser.scaffold_module.enabled=false \
    denoiser.dit.scaffolding.use_concat=true \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    train.batch_size=64 \
    train.compile_model=true \
    hydra.run.dir=${HYDRA_DIR}
