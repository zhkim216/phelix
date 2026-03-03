#!/bin/bash
#SBATCH --job-name=run_tc_eval_exp18_cfg456
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-149
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/run_tc_eval_exp18_cfg456_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/run_tc_eval_exp18_cfg456_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# ===== Parallelization settings =====
NUM_DATA_CHUNKS=50  # Number of data-parallel chunks per combination

COMBINATIONS=(
    "exp18_cfg4_step_42500,rfd3val,false"
    "exp18_cfg5_step_42500,rfd3val,false"
    "exp18_cfg6_step_42500,rfd3val,false"
)
# Format: "MODEL_NAME,VALSET_NAME,PROTEIN_ONLY"
# Add more combinations as needed, e.g.:
# "caliby,lmpnnval,false"

# 2D indexing: combo x data_chunk
NUM_COMBOS=${#COMBINATIONS[@]}
TOTAL_TASKS=$(( NUM_COMBOS * NUM_DATA_CHUNKS ))
# Note: set --array=0-$((TOTAL_TASKS - 1)) in the SBATCH header above

COMBO_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_DATA_CHUNKS ))
DATA_CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_DATA_CHUNKS ))

# Parse the current combination
IFS=',' read -r MODEL_NAME VALSET_NAME PROTEIN_ONLY <<< "${COMBINATIONS[$COMBO_IDX]}"

# Input paths (flat directory)
PDB_DIRS=(
    "/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp18_cfg4_rfd3val_af3/step_52500_epoch_33/samples"
    "/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp18_cfg5_rfd3val_af3/step_52500_epoch_33/samples"
    "/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp18_cfg6_rfd3val_af3/step_52500_epoch_33/samples"
    

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3_tc

echo "Starting TC eval for model=${MODEL_NAME}, valset=${VALSET_NAME}, protein_only=${PROTEIN_ONLY}, data_chunk=${DATA_CHUNK_IDX}/${NUM_DATA_CHUNKS}"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_tc_eval_af3.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    protein_only=${PROTEIN_ONLY} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=null \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    'pdb_cfg.sample_indices=[0,1,2,3,4]' \
    pdb_cfg.array_id=${DATA_CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_DATA_CHUNKS} \
    pocket_cfg.pocket_distance_for_metrics=6.0 \
    struct_pred_cfg.evaluate_docking_consistency=true \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=1 \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "TC eval complete for model=${MODEL_NAME}, data_chunk=${DATA_CHUNK_IDX}"
