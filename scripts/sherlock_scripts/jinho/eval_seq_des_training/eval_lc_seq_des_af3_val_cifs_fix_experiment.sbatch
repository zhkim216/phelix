#!/bin/bash
#SBATCH --job-name=eval_af3_val_cifs_fix_experiment
#SBATCH -p possu,bioe
#SBATCH --array=0-4
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
GROUP=eval_af3_exp2

# Experiment names 
EXP_NAME_BASE="exp2_cfg5"
EXP_NAME="eval_${EXP_NAME_BASE}_af3_val_cifs"

# Define checkpoint steps (10 checkpoints, 2 per group)
CHECKPOINTS=(2500 12500 22500 32500 42500 52500 62500 72500 82500 92500)

# Array job mapping:
# 0-4: checkpoint groups 0-4, each with 2 checkpoints
# fix_pocket_seq is fixed to true

FIX_POCKET_SEQ=true
CKPT_GROUP=$SLURM_ARRAY_TASK_ID

# Get the two checkpoint steps for this group (each group has 2 checkpoints)
CKPT_IDX_1=$((CKPT_GROUP * 2))
CKPT_IDX_2=$((CKPT_GROUP * 2 + 1))
CKPT_STEP_1=${CHECKPOINTS[$CKPT_IDX_1]}
CKPT_STEP_2=${CHECKPOINTS[$CKPT_IDX_2]}

echo "========================================"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "EXP_NAME_BASE: $EXP_NAME_BASE"
echo "FIX_POCKET_SEQ: $FIX_POCKET_SEQ"
echo "CKPT_GROUP: $CKPT_GROUP"
echo "CHECKPOINT_STEPS: $CKPT_STEP_1, $CKPT_STEP_2"
echo "========================================"

# Input paths
CKPT_DIR="/scratch/users/zhkim216/out_dir/ligand_seq_des/${EXP_NAME_BASE}"
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_af3_valset/interface_df_for_fix_exp.parquet
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_af3_valset/cached_examples
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/af3_val_cifs/pdbs
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data

# Check if checkpoint directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

echo "Checkpoint directory: $CKPT_DIR"
echo "Starting evaluation..."

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/eval_lc_seq_des_training.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${GROUP} \
    seed=0 \
    debug=false \
    start_step=${CKPT_STEP_1} \
    end_step=${CKPT_STEP_2} \
    eval_every_n_ckpts=4 \
    eval_last_ckpt=false \
    use_ema=true \
    evaluate_self_consistency=false \
    evaluate_template_conditioned_docking=true \
    protein_only=false \
    fix_pocket_seq=${FIX_POCKET_SEQ} \
    pocket_distance=8.0 \
    skip_sampling=false \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des \
    data_cfg_for_design.load_cache_cfg.cached_example_path=${CACHED_EXAMPLE_PATH} \
    data_cfg_for_design.pdb_cfg.pdb_dir=${PDB_DIR} \
    transform_cfg_for_design.featurizer_cfg.residue_cache_dir=${RESIDUE_CACHE_DIR} \
    denoiser_train_dir=${CKPT_DIR} \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=1 \
    sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=500 \
    sampling_cfg_overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=1 \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=5 \
    docking_metrics_cfg.pocket_distance_for_align=8.0 \
    hydra.run.dir=${HYDRA_DIR}

echo "All evaluations complete for fix_pocket_seq=${FIX_POCKET_SEQ}, checkpoints ${CKPT_STEP_1} and ${CKPT_STEP_2}"
