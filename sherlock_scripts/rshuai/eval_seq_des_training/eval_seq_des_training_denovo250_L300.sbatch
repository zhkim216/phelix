#!/bin/bash
#SBATCH --job-name=aasd1_a1_v1_eval_seq_des
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and set environment variables
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/users/rshuai/model_params/af2

# Inputs config
PDB_DIR=/scratch/users/rshuai/datasets/denovo250dev/pdbs

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to evaluate
MODEL_NAME=aasd1_a1_v1
DENOISER_TRAIN_DIR=/scratch/users/rshuai/allatom_design/allatom_design/${MODEL_NAME}
NOISE_LABELS=0.5

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_eval_seq_des_denovo250dev_L300_nl${NOISE_LABELS}_af2_AR
GROUP=seq_des

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_seq_des_training.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.model_name=af2 \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.subset_length_range=[300,300] \
    start_step=0 \
    end_step=null \
    eval_every_n_ckpts=1 \
    sampling_cfg_overrides.noise_labels=${NOISE_LABELS} \
    +sampling_cfg_overrides.use_potts_sampling=false \
    hydra.run.dir=${HYDRA_DIR}
