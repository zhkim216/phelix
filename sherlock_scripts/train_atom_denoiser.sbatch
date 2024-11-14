#!/bin/bash
#SBATCH --job-name=atom_denoiser_baseline
#SBATCH -p possu,bioe
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
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/subhang/allatom-design
HYDRA_DIR=/scratch/users/rshuai/subhang/hydra_outputs
OUT_DIR=/scratch/users/rshuai/subhang/allatom_design

cd ${BASE_DIR}

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/subhang/cache/torch

# Wandb settings / experiment name
WANDB_ID=null
EXP_NAME=atom_denoiser_baseline
GROUP=edm2_v2

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/users/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.fixed_size=256 \
    train.batch_size=64 \
    train.compile_model=false \
    trainer.strategy=ddp_find_unused_parameters_true \
    model.task=backbone \
    denoiser=dit_denoiser \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=80 \
    checkpointing.save_ema_every_n_epochs=20 \
    trainer.check_val_every_n_epoch=20 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
