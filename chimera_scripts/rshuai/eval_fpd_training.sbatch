#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Define directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
FPD_EMBEDDINGS_DIR=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/fpd_embeddings
ANNOTATION_CSV=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations_for_eval2/annotated_pdb_keys.csv

# Set HF variables
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Model name to evaluate
MODEL_NAME=u1_a1_phema
EXP_NAME=${MODEL_NAME}_eval_fpd_training
WANDB_ID=richardshuai
GROUP=eval_fpd_training

# Set HF variables
export HUGGINGFACE_TOKEN=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
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_fpd_training.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    fpd_embeddings_dir=${FPD_EMBEDDINGS_DIR} \
    annotation_csv=${ANNOTATION_CSV} \
    use_esm3=true \
    start_step=0 \
    eval_every_n_ckpts=1 \
    batch_size=64 \
    num_steps=500 \
    pct_subsample_train=0.1 \
    max_scrmsd=2.0 \
    max_rel_rog=1.4 \
    subset_length_range=[32,256] \
    hydra.run.dir=${HYDRA_DIR}
