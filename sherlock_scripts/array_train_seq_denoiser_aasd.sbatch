#!/bin/bash
#SBATCH --job-name=a8_aasd0_eps
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-9%10                           # 10 tasks with max 10 concurrent jobs
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
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
PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/users/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Define eps values for the sweep
eps_values=(0.0 0.01 0.02 0.1 0.2 0.3 0.4 0.5 0.6 1.0)

# Retrieve the specific eps value for this task
eps=${eps_values[$SLURM_ARRAY_TASK_ID]}

EXP_NAME=a8_aasd0_eps_${eps}
GROUP=chimera_a8

# Run training script with the selected eps
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=128 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    train.batch_size=64 \
    train.compile_model=true \
    model.task=allatom_seq_des \
    interpolant.training_noise_schedule=uniform_squared_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.augment_eps=${eps} \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=20 \
    trainer.check_val_every_n_epoch=20 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
