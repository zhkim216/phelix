#!/bin/bash
#SBATCH --job-name=if_benchmark
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err

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
GROUP=denovo250dev_if_benchmarking

cd ${BASE_DIR}

# Run inverse folding script
MODEL_NAME=c2_aasd2
BASE_NAME=denovo250dev_S100_so_nl0.3
EXP_NAME=${MODEL_NAME}_${BASE_NAME}
INPUT_PDB_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/sample_multi/ema-step300000-std0.220/${BASE_NAME}/samples
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/inverse_fold_benchmarking.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_PER_TASK} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    sample_dir=/scratch/hielab/rshuai/datasets/denovo250dev/pdbs \
    input_pdb_dir=${INPUT_PDB_DIR} \
    struct_pred_cfg.model_name=af2 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/inverse_fold_benchmarking/ema-step300000-std0.220/${EXP_NAME}
