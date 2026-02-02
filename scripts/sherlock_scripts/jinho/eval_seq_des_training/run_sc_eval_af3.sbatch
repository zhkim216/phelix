#!/bin/bash
#SBATCH --job-name=run_sc_eval_af3
#SBATCH -p bioe,possu
#SBATCH --array=0-19
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Input paths
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered_re/interface_df_full.parquet

# 4 distances x 5 models = 20 combinations
POCKET_DISTANCES=(4.0 5.0 6.0 8.0)
MODEL_NAMES=("pmpnn" "lmpnn005" "lmpnn010" "lmpnn020" "lmpnn030")

# Calculate indices from SLURM_ARRAY_TASK_ID
DISTANCE_IDX=$((SLURM_ARRAY_TASK_ID % 4))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 4))

POCKET_DISTANCE=${POCKET_DISTANCES[$DISTANCE_IDX]}
MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}

# Exp names and output paths
EXP_NAME=caliby_wocenter_lmpnnval_redesign_${MODEL_NAME}_${POCKET_DISTANCE}
PDB_DIR="${BASE_OUT_DIR}/${EXP_NAME}/samples/backbones"

echo "=========================================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "POCKET_DISTANCE: ${POCKET_DISTANCE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "EXP_NAME: ${EXP_NAME}"
echo "PDB_DIR: ${PDB_DIR}"
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
    cached_example_path=${CACHED_EXAMPLE_PATH} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"

