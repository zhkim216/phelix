#!/bin/bash
#SBATCH --job-name=ligandmpnn_sc_eval
#SBATCH -p bioe,possu
#SBATCH --array=0
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
## Write batch script to logs
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

# Model params
STRUCT_PRED_MODEL=boltz1
BOLTZ1_CKPT=/scratch/users/rshuai/model_params/boltz1/boltz1_conf.ckpt

# Set cache directories
export HF_HOME=/scratch/users/rshuai/cache/huggingface
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Inputs config
PDB_DIR=/scratch/users/rshuai/datasets/boltz_test_cifs/pdbs
PDB_LIST_STEM=protein_monomer_32_512_no_ligand
PDB_NAME_LIST=/scratch/users/rshuai/datasets/boltz_test_cifs/pdb_name_lists/${PDB_LIST_STEM}.txt
EVALS_BASE_DIR=/scratch/users/rshuai/allatom_design/allatom_design/boltz_test_cifs_evals/${PDB_LIST_STEM}

# Model to evaluate
EVAL_NAMES=(
    "ligandmpnn"
)
EVAL_NAME=${EVAL_NAMES[$SLURM_ARRAY_TASK_ID]}
LIGANDMPNN_OUTPUTS_DIR=${EVALS_BASE_DIR}/${EVAL_NAME}/designs

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${EVAL_NAME}_${PDB_LIST_STEM}_${STRUCT_PRED_MODEL}
GROUP=seq_des
BASE_OUT_DIR=${EVALS_BASE_DIR}/${EVAL_NAME}

# Run script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/benchmarking/ligandmpnn_sc_eval.py \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    exp_name=${EXP_NAME} \
    struct_pred_cfg.model_name=${STRUCT_PRED_MODEL} \
    struct_pred_cfg.boltz1.checkpoint=${BOLTZ1_CKPT} \
    base_out_dir=${BASE_OUT_DIR} \
    ligandmpnn_outputs_dir=${LIGANDMPNN_OUTPUTS_DIR} \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.pdb_name_list=${PDB_NAME_LIST} \
    input_cfg.skip_pdb_names="['8add.cif', '8qcw.cif']" \
    max_samples_per_pdb=8 \
    hydra.run.dir=${HYDRA_DIR}
