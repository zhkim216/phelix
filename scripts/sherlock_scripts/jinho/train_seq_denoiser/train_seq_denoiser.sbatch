#!/bin/bash
#SBATCH --job-name=lpotts_test_4
#SBATCH -p possu
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%j.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

source ~/.bashrc

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.4
venv enzyme_design

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

# Set cache directories
export TORCH_HOME=/scratch/users/zhkim216/cache/torch
export HF_HOME=/scratch/users/zhkim216/cache/huggingface

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP=""
EXP_NAME=uncond_potts_bftokenbugs

# Dataset paths
PDB_PATH=/scratch/users/zhkim216/datasets/boltz_v2

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    --config-dir allatom_design/configs/seq_denoiser \
    --config-name seq_denoiser.yaml \
    wandb.wandb_id=${WANDB_ID} \
    wandb.project=${PROJECT} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    num_workers=${SLURM_CPUS_PER_TASK} \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    data.pdb_path=${PDB_PATH} \
    data.max_tokens=512 \
    data.max_atoms=4608 \
    data.sampler.beta_chain=1.0 \
    data.sampler.beta_interface=1.0 \
    data.cropper.only_keep_resolved=true \
    data.cropper.max_neighborhood=100 \
    train.batch_size=32 \
    trainer.accumulate_grad_batches=1 \
    train.compile_model=true \
    mask_selector.restype_masking_schedule=uniform_t \
    denoiser.mpnn.use_atom_encoder=false \
    denoiser.mpnn.token_features.ca_only=false \
    denoiser.mpnn.n_layers=3 \
    denoiser.augment_eps=0.3 \
    data.filters.4.max_residues=2048 \
    data.cropper._target_=allatom_design.data.crop.boltz.BoltzCropper \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    ${OVERRIDES} \
    hydra.run.dir=${HYDRA_DIR}