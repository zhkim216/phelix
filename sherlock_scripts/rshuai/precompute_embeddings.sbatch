#!/bin/bash
#SBATCH --job-name=pc_esm3_embs
#SBATCH -p possu,bioe
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
BASE_DIR=/home/users/rshuai/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Set up Hugging Face
export HUGGINGFACE_TOKEN=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
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Model params
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn

# Directories
TRAIN_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/train_pdb_keys.list
EVAL_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/eval_pdb_keys.list
EVAL2_PDB_KEY_FILE=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/eval2_pdb_keys.list

PDBS_DIR=/scratch/users/rshuai/subhang/datasets/augmented_ingraham_cath_bugfree/all_pdbs

MPNN_OUT_DIR=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/fpd_embeddings/mpnn
ESM3_OUT_DIR=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree/fpd_embeddings/esm3

# Run eval script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/precompute_embeddings.py \
    compute_mpnn=false \
    compute_esm3=true \
    train_pdb_key_file=${TRAIN_PDB_KEY_FILE} \
    eval_pdb_key_file=${EVAL_PDB_KEY_FILE} \
    eval2_pdb_key_file=${EVAL2_PDB_KEY_FILE} \
    pdbs_dir=${PDBS_DIR} \
    mpnn.out_dir=${MPNN_OUT_DIR} \
    esm3.out_dir=${ESM3_OUT_DIR} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    hydra.run.dir=${HYDRA_DIR}
