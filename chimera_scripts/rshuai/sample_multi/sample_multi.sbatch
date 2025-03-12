#!/bin/bash
#SBATCH --job-name=sample_multi
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Set paths
MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c1_aasd4_eps0.3/checkpoints/sd-step150000-epoch450.ckpt
PDB_DIR=/scratch/hielab/rshuai/datasets/denovo250dev/pdbs
OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c1_aasd4_eps0.3/sample_multi/sd-step150000-epoch450/denovo250dev_S100_so_nl0.3

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_PER_TASK} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    out_dir=${OUT_DIR} \
    noise_labels=0.3 \
    num_steps=100 \
    seq_only=true \
    temperature=0.1
