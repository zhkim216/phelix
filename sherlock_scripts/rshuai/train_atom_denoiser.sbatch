#!/bin/bash
#SBATCH --job-name=s0_bb3
#SBATCH -p bioe
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
WANDB_ID=richardshuai
cd ${BASE_DIR}

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Set experiment name and grouop
EXP_NAME=s0_bb3
GROUP=ai_cath

# Dataset paths
# PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset
PDB_PATH=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_paths=[${PDB_PATH}] \
    data.fixed_size=160 \
    data.subset_length_range=[32,160] \
    model.task=scaffold \
    scaffold_manager=scaffold_manager \
    denoiser.scaffold_module.enabled=true \
    train.batch_size=16 \
    denoiser.training_batch_size_mult=4 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    trainer.check_val_every_n_epoch=1 \
    trainer.strategy=ddp_find_unused_parameters_true \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}