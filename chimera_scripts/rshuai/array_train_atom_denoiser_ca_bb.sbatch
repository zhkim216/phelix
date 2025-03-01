#!/bin/bash
#SBATCH --job-name=a5_cbb_array
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-24
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#

## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
WANDB_ID=richardshuai
cd $BASE_DIR

EXP_NAME=a5_cbb${SLURM_ARRAY_TASK_ID}
GROUP=chimera_a5

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset

# Define parameter grids
PSIGMA_MEANS=(-1.0 -0.5 -0.5 0 1.0)
PSIGMA_STDS=(1.0 1.0 1.5 1.5 1.5)

# Calculate indices for `ca` and `nco` based on SLURM_ARRAY_TASK_ID
TASK_ID=$SLURM_ARRAY_TASK_ID

CA_INDEX=$(( TASK_ID / 5 ))  # Integer division
NCO_INDEX=$(( TASK_ID % 5 ))  # Modulo operation

CA_PSIGMA_MEAN=${PSIGMA_MEANS[$CA_INDEX]}
CA_PSIGMA_STD=${PSIGMA_STDS[$CA_INDEX]}

NCO_PSIGMA_MEAN=${PSIGMA_MEANS[$NCO_INDEX]}
NCO_PSIGMA_STD=${PSIGMA_STDS[$NCO_INDEX]}

# Run training script with the selected parameters
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_path=${PDB_PATH} \
    data.bb_only=false \
    train.batch_size=64 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    data.fixed_size=128 \
    model.task=ca_backbone \
    model.use_self_conditioning=true \
    interpolant=multimodal/edm_ca_interpolant \
    interpolant.ca.training_noise_schedule=lognormal \
    interpolant.ca.training_noise_cfg.lognormal.psigma_mean=${CA_PSIGMA_MEAN} \
    interpolant.ca.training_noise_cfg.lognormal.psigma_std=${CA_PSIGMA_STD} \
    interpolant.nco.training_noise_schedule=lognormal \
    interpolant.nco.training_noise_cfg.lognormal.psigma_mean=${NCO_PSIGMA_MEAN} \
    interpolant.nco.training_noise_cfg.lognormal.psigma_std=${NCO_PSIGMA_STD} \
    eval.eval_timesteps=[] \
    eval.mm_eval.fix_t1=[0.0,0.2,0.4,0.6,0.8,1.0] \
    eval.mm_eval.mm_eval_timesteps=[0.0,0.2,0.4,0.6,0.8,1.0] \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_epochs=20 \
    trainer.check_val_every_n_epoch=20 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
