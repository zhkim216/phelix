#!/bin/bash
#SBATCH --job-name=pc_fampnn_embs
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Set up Hugging Face
export HUGGINGFACE_TOKEN=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
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn
FAMPNN_MODEL_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt

# Directories
## if we need to subset the original pdb key list
# TRAIN_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/train_pdb_keys.list
# EVAL_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/eval_pdb_keys.list
# EVAL2_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/eval2_pdb_keys.list
# USE_PRECOMPUTED_KEY_FILES=false

## if we've already subsetted the pdbs
# TRAIN_PDB_KEY_FILE=null
# EVAL_PDB_KEY_FILE=null
# EVAL2_PDB_KEY_FILE=null
# USE_PRECOMPUTED_KEY_FILES=true

# PDBS_DIR=/scratch/users/rshuai/datasets/augmented_ingraham_cath_bugfree/all_pdbs
# OUT_DIR=/scratch/users/rshuai/datasets/augmented_ingraham_cath_bugfree/fampnn_embeddings

TRAIN_PDB_KEY_FILE=/scratch/users/rshuai/datasets/af3_pdb_monomer/train_pdb_keys_for_eval2.list
EVAL_PDB_KEY_FILE=/scratch/users/rshuai/datasets/af3_pdb_monomer/eval_pdb_keys_for_eval2.list
EVAL2_PDB_KEY_FILE=/scratch/users/rshuai/datasets/af3_pdb_monomer/eval2_pdb_keys_for_eval2.list
USE_PRECOMPUTED_KEY_FILES=false

PDB_PATH=/scratch/users/rshuai/datasets/af3_pdb_monomer
OUT_DIR=/scratch/users/rshuai/datasets/af3_pdb_monomer/fpd_embeddings

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/precompute_embeddings.py \
    compute_mpnn=false \
    compute_esm3=true \
    compute_fampnn=false \
    use_precomputed_key_files=${USE_PRECOMPUTED_KEY_FILES} \
    train_pdb_key_file=${TRAIN_PDB_KEY_FILE} \
    eval_pdb_key_file=${EVAL_PDB_KEY_FILE} \
    eval2_pdb_key_file=${EVAL2_PDB_KEY_FILE} \
    pdb_path=${PDB_PATH} \
    out_dir=${OUT_DIR} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    fampnn.checkpoint_path=${FAMPNN_MODEL_CKPT} \
    subsample_n=null \
    hydra.run.dir=${HYDRA_DIR}
