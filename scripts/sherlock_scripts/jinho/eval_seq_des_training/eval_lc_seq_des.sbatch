#!/bin/bash
#SBATCH --job-name=eval_af3
#SBATCH -p bioe,possu
#SBATCH --array=0-7
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Experiment names
# Array 0-5: standard experiments (protein_only=false)
# Array 6-7: protein_only=true for exp2_cfg2, exp2_cfg5
EXP_NAMES=("exp2_cfg0" "exp2_cfg1" "exp2_cfg2" "exp2_cfg3" "exp2_cfg4" "exp2_cfg5" "exp2_cfg2" "exp2_cfg5")

EXP_NAME_BASE=${EXP_NAMES[SLURM_ARRAY_TASK_ID]}

# Set protein_only flag based on array index
if [[ $SLURM_ARRAY_TASK_ID -eq 6 || $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
    PROTEIN_ONLY=true
    EXP_NAME=eval_af3_${EXP_NAME_BASE}_protein_only
else
    PROTEIN_ONLY=false
    EXP_NAME=eval_af3_${EXP_NAME_BASE}
fi

echo "========================================"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "EXP_NAME_BASE: $EXP_NAME_BASE"
echo "EXP_NAME: $EXP_NAME"
echo "PROTEIN_ONLY: $PROTEIN_ONLY"
echo "========================================"

# Input paths
CKPT_DIR="/scratch/users/zhkim216/out_dir/ligand_seq_des/${EXP_NAME_BASE}"
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered_re/interface_df_rep_cluster.parquet

# Check if checkpoint directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

echo "Checkpoint directory: $CKPT_DIR"
echo "Starting evaluation..."

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_lc_seq_des_training.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=eval \
    seed=0 \
    debug=false \
    start_step=null \
    end_step=null \
    use_ema=true \
    protein_only=${PROTEIN_ONLY} \
    evaluate_self_consistency=false \
    evaluate_template_conditioned_docking=true \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des \
    input_cfg.load_from_cache=true \
    input_cfg.load_cache_cfg.cached_example_path=${CACHED_EXAMPLE_PATH} \
    denoiser_train_dir=${CKPT_DIR} \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=1 \
    sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=500 \
    sampling_cfg_overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"

