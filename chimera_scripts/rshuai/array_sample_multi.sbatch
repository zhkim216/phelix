#!/bin/bash
#SBATCH --job-name=c0_aasd2_val_sample_multi
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-4%2

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

# Array values for num_steps
NUM_STEPS=100
PSCE_ARRAY=(0.1 0.2 0.3 0.5 1.0)
PSCE_THRESHOLD=${PSCE_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Set paths
MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.010.ckpt
OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/sample_multi/ema-step300000-std0.010/val_S${NUM_STEPS}_psce${PSCE_THRESHOLD}
PDB_DIR=/scratch/hielab/talal/datasets/af3_pdb/cached_examples
PDB_KEY_LIST=/scratch/hielab/talal/datasets/af3_pdb/val_pdb_keys_L64_512_single_chain.list

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    pdb_key_ext=.pt \
    from_cache=true \
    out_dir=${OUT_DIR} \
    num_steps=${NUM_STEPS} \
    psce_threshold=${PSCE_THRESHOLD} \
    repack_every_step=false \
    temperature=0.1
