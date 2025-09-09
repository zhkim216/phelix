#!/bin/bash
#SBATCH --job-name=seq_des_multi
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
module load cuda/12.4
ENV_DIR=/home/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories and paths
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
SOFTWARE_PATH=/oak/stanford/groups/possu/rshuai/software  # e.g. for MASTER, redis
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.

# Model params (if running self-consistency eval)
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Inputs config
PDB_DIR=/scratch/users/rshuai/datasets/boltz_val_cifs/pdbs
PDB_NAME_LIST=/scratch/users/rshuai/datasets/boltz_val_cifs/pdb_name_lists/debug.txt
POS_CONSTRAINT_CSV=/scratch/users/rshuai/datasets/boltz_val_cifs/fixed_pos_csvs/debug.csv

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Model to use
CKPT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/sd3_a9_S/checkpoints/ema/sd-step25000-epoch15-ema0.99.ckpt
OUT_DIR=/scratch/users/rshuai/test_seq_des_multi

# Run training script
cd ${BASE_DIR}
python3 allatom_design/eval/sampling/seq_des_multi.py \
    ckpt_path=${CKPT_PATH} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    run_self_consistency_eval=false \
    sampling_cfg_overrides.batch_size=16 \
    sampling_cfg_overrides.num_seqs_per_pdb=3 \
    base_out_dir=${OUT_DIR} \
    hydra.run.dir=${HYDRA_DIR}
