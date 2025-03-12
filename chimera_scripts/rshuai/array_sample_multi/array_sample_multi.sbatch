#!/bin/bash
#SBATCH --job-name=array_sample_multi
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-24%2

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

# Array values, help me get EPS and NOISE_LABELS
EPS_VALUES=(0.0 0.1 0.2 0.3 0.4)
NOISE_LABELS_VALUES=(0.0 0.1 0.2 0.3 0.4)
INDEX=${SLURM_ARRAY_TASK_ID}
EPS_INDEX=$((INDEX / 5))
NOISE_INDEX=$((INDEX % 5))
EPS=${EPS_VALUES[$EPS_INDEX]}
NOISE_LABELS=${NOISE_LABELS_VALUES[$NOISE_INDEX]}

NUM_STEPS=100
PSCE_THRESHOLD=null

# Set paths
# MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.010.ckpt
# OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c0_aasd2/sample_multi/ema-step300000-std0.010/val_S${NUM_STEPS}_psce${PSCE_THRESHOLD}
# PDB_DIR=/scratch/hielab/talal/datasets/af3_pdb/cached_examples
# PDB_KEY_LIST=/scratch/hielab/talal/datasets/af3_pdb/val_pdb_keys_L64_512_single_chain.list

MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c1_aasd2_eps${EPS}/checkpoints/sd-step150000-epoch450.ckpt
OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c1_aasd2_eps${EPS}/sample_multi/sd-step150000-epoch450/denovo250dev_S${NUM_STEPS}_so_nl${NOISE_LABELS}
PDB_DIR=/scratch/hielab/rshuai/datasets/denovo250dev/pdbs

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/sample_multi.py \
    num_workers=${SLURM_CPUS_PER_TASK} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    out_dir=${OUT_DIR} \
    num_steps=${NUM_STEPS} \
    psce_threshold=${PSCE_THRESHOLD} \
    noise_labels=${NOISE_LABELS} \
    seq_only=true \
    temperature=0.1