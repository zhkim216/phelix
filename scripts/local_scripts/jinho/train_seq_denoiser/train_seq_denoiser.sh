#!/bin/bash


source ~/.bashrc

## Load modules and conda environment
venv enzyme_design

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/allatom_design

# Set cache directories
export TORCH_HOME=/scratch/users/zhkim216/cache/torch
export HF_HOME=/scratch/users/zhkim216/cache/huggingface

# Wandb settings / experiment name
EXP_NAME=protein_potts_test
GROUP=seq_des

# Dataset paths
PDB_PATH=/scratch/users/zhkim216/datasets/boltz_v2

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
    data.cropper._target_=allatom_design.data.crop.boltz.BoltzCropper \
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
    loss.seq_loss.label_smoothing=0.157 \
    loss.potts.label_smoothing=0.157 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR}