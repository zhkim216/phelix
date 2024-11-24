#!/bin/bash
#SBATCH --job-name=b9_aasd4_mlp_dropout
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
WANDB_ID=richardshuai
cd $BASE_DIR

# Define array for mlp_dropout
mlp_dropout_values=(0.1 0.2)

# Calculate indices
total_jobs=${#mlp_dropout_values[@]}

mlp_dropout_idx=$SLURM_ARRAY_TASK_ID

# Get the current mlp_dropout based on index
MLP_DROPOUT=${mlp_dropout_values[$mlp_dropout_idx]}

# Set experiment name and group, including mlp_dropout
EXP_NAME=b9_aasd4_mlp_dropout${MLP_DROPOUT}
GROUP=chimera_b9

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Run training script with the selected mlp_dropout
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.fixed_size=256 \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    data.subset_length_range=[16,9999] \
    train.batch_size=64 \
    train.compile_model=true \
    model.task=allatom_seq_des \
    interpolant.training_noise_schedule=uniform_sqrt_t \
    denoiser=minimpnn_denoiser \
    denoiser.minimpnn.model_type=sidechain \
    denoiser.minimpnn.augment_eps=0.3 \
    denoiser.scn_diffusion_module.training_batch_size_mult=8 \
    denoiser.scn_diffusion_module.hidden_size=256 \
    denoiser.scn_diffusion_module.scn_denoiser.depth=4 \
    denoiser.scn_diffusion_module.scn_denoiser.mlp_dropout=${MLP_DROPOUT} \
    denoiser.scn_diffusion_module.scn_denoiser.mlp_ratio=4.0 \
    optim.optimizer=noam \
    checkpointing.save_latest_every_n_steps=5000 \
    checkpointing.save_ema_every_n_steps=1500 \
    trainer.check_val_every_n_epoch=10 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
