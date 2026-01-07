#!/bin/bash
#SBATCH --job-name=run_sc_eval_caliby_redesign_lmpnn_rfd3val_af3
#SBATCH -p possu
#SBATCH --array=0-11
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${BASE_OUT_DIR}"

# 3 distances x 4 models = 12 combinations
POCKET_DISTANCES=(4.0 5.0 6.0)
MODEL_NAMES=("lmpnn005" "lmpnn010" "lmpnn020" "lmpnn030")

# Calculate indices from SLURM_ARRAY_TASK_ID
DISTANCE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 3))

POCKET_DISTANCE=${POCKET_DISTANCES[$DISTANCE_IDX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Input paths
METADATA_PATH=null
PDB_DIR=${BASE_OUT_DIR}/eval_caliby_${MODEL_NAME}_redesign_${POCKET_DISTANCE}_rfd3val_af3/samples
PDB_NAME_LIST=null
PDB_NAME_EXT=.pdb
LIGAND_SOURCE_PATH=null

# Exp name
EXP_NAME=eval_caliby_${MODEL_NAME}_redesign_${POCKET_DISTANCE}_rfd3val_af3

echo "=========================================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "POCKET_DISTANCE: ${POCKET_DISTANCE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "PDB_DIR: ${PDB_DIR}"
echo "EXP_NAME: ${EXP_NAME}"
echo "=========================================="
echo "Starting evaluation..."

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_sc_eval_af3.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    seed=0 \
    debug=false \
    calculate_metrics_only=false \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=${PDB_NAME_EXT} \
    ligand_source_cfg.ligand_source_path=${LIGAND_SOURCE_PATH} \
    ligand_source_cfg.add_ligands_to_designed_samples=false \
    ligand_source_cfg.source_is_designed=true \
    is_all_atom_sample=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"