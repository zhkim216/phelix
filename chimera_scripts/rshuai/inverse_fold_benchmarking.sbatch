#!/bin/bash
#SBATCH --job-name=if_benchmark
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=0-7

MODELS=("esm3" "esm_if1" "lmdesign" "proteinmpnn")
MODEL_ID=$((SLURM_ARRAY_TASK_ID % 4))
MODEL_NAME=${MODELS[$MODEL_ID]}

# Determine if we are running the original or esmfold variant
if [ ${SLURM_ARRAY_TASK_ID} -lt 4 ]; then
    # Original run
    EXP_NAME=${MODEL_NAME}
    ESMFOLD_FLAG=""
    OUTPUT_SUFFIX=""
else
    # ESMfold run
    EXP_NAME=${MODEL_NAME}_esmfold
    ESMFOLD_FLAG="struct_pred_cfg.model_name=esmfold"
    OUTPUT_SUFFIX="_esmfold"
fi

# For proteinmpnn, input_fasta_dir should be null
if [ "${MODEL_NAME}" == "proteinmpnn" ]; then
    INPUT_FASTA_DIR="null"
else
    INPUT_FASTA_DIR="/scratch/hielab/rshuai/allatom_design/allatom_design/sc_evals/fastas/${MODEL_NAME}"
fi

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Wandb settings / experiment name
WANDB_ID=richardshuai
GROUP=denovo500_if_benchmarking

cd ${BASE_DIR}

# Run inverse folding script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/inverse_fold_benchmarking.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    sample_dir=/scratch/hielab/rshuai/datasets/denovo500/pdbs \
    input_fasta_dir=${INPUT_FASTA_DIR} \
    ${ESMFOLD_FLAG} \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=/scratch/hielab/rshuai/allatom_design/allatom_design/sc_evals/outputs/${MODEL_NAME}${OUTPUT_SUFFIX}
