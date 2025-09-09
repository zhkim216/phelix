#!/bin/bash
#SBATCH --job-name=run_sc_eval
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
MPNN_PARAMS_DIR=/scratch/hielab/rshuai/model_params/mpnn
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Inputs config
PDB_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c6_aasd1/eval_seq_des_training/c6_aasd1_eval_seq_des_nl0.1/step_120000_epoch_344/samples
MODEL_NAME=esmfold

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=c6_aasd1_eval_seq_des_nl0.1_${MODEL_NAME}
GROUP=seq_des
BASE_OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/chroma_evals/af3_pdb_monomer

# Run training script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_sc_eval.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    seq_des_cfg.proteinmpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.model_name=${MODEL_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.subset_length_range=[32,256] \
    input_cfg.n_subsample=1024 \
    use_seq_design=false \
    hydra.run.dir=${HYDRA_DIR}
