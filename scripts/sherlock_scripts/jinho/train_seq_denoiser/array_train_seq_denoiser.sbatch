#!/bin/bash
#SBATCH --job-name=lpotts_env_test
#SBATCH -p possu
#SBATCH --array=0-2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00
#
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%j.err
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.bashrc
module load gcc/12.4.0
module load cuda/12.6.1
mamba activate ppg

# Enforce glibc 2.38 (torch 2.7 & boltz2x)
export RUN_GLIBC238_REQUIRE_CUDA=1

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
GROUP="env_test"
EXP_NAME_BASE="env_test"

# Dataset paths
PDB_PATH=/scratch/users/zhkim216/datasets/boltz_v2

# Experiment configs
DENOISER_TYPE="ligand_potts"

CONFIGS=(
"train.compile_model=false denoiser.mpnn.use_atom_encoder=true mask_selector.atom_masking_schedule=uniform_t denoiser.mpnn.atom_encoder.k_atom_neighbors=24 denoiser.mpnn.atom_encoder.n_layers=2"
"train.compile_model=true train.compile_backend=aot_eager denoiser.mpnn.use_atom_encoder=true mask_selector.atom_masking_schedule=uniform_t denoiser.mpnn.atom_encoder.k_atom_neighbors=24 denoiser.mpnn.atom_encoder.n_layers=2"
"train.compile_model=true train.compile_backend=inductor denoiser.mpnn.use_atom_encoder=true mask_selector.atom_masking_schedule=uniform_t denoiser.mpnn.atom_encoder.k_atom_neighbors=24 denoiser.mpnn.atom_encoder.n_layers=2"
)

OVERRIDES="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAME_BASE}_cfg${SLURM_ARRAY_TASK_ID}"

# Run model
cd "${BASE_DIR}"
WANDB__SERVICE_WAIT=300 python allatom_design/train_seq_denoiser.py \
    --config-dir allatom_design/configs/seq_denoiser \
    --config-name seq_denoiser.yaml \
    wandb.wandb_id=${WANDB_ID} \
    wandb.project=${PROJECT} \
    wandb.group=${GROUP} \
    wandb.no_wandb=false \
    num_workers=${SLURM_CPUS_PER_TASK} \
    out_dir=${OUT_DIR} \
    exp_name=${EXP_NAME} \
    train.batch_size=8 \
    trainer.accumulate_grad_batches=4 \
    data.pdb_path=${PDB_PATH} \
    data.max_tokens=512 \
    data.max_atoms=4608 \
    data.filters.4.max_residues=2048 \
    data.cropper._target_=allatom_design.data.crop.boltz.BoltzCropper \
    data.sampler.alpha_prot=3.0 \
    data.sampler.alpha_ligand=1.0 \
    data.sampler.beta_chain=1.0 \
    data.sampler.beta_interface=1.0 \
    data.cropper.only_keep_resolved=true \
    data.cropper.max_neighborhood=100 \
    mask_selector.restype_masking_schedule=uniform_t \
    denoiser.mpnn.token_features.ca_only=false \
    denoiser.mpnn.token_features.token_bonds_masking=true \
    denoiser.mpnn.n_layers=3 \
    denoiser.per_residue_eps=false \
    denoiser.augment_eps=0.3 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    ${OVERRIDES} \
    hydra.run.dir=${HYDRA_DIR}
