#!/bin/bash
#SBATCH --job-name=eval_af3
#SBATCH -p possu,bioe
#SBATCH --array=0-2
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
EXP_NAME_BASE="exp2_cfg2"
EXP_NAME="eval_af3_${EXP_NAME_BASE}"

# Pocket distances
POCKET_DISTANCES=(5.0 6.5 8.0)
POCKET_DISTANCE=${POCKET_DISTANCES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "EXP_NAME_BASE: $EXP_NAME_BASE"
echo "EXP_NAME: $EXP_NAME"
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
    wandb.group=${GROUP} \
    seed=0 \
    debug=false \
    start_step=82500 \
    end_step=82500 \
    eval_every_n_ckpts=4 \
    eval_last_ckpt=false \
    use_ema=true \
    evaluate_self_consistency=true \
    evaluate_template_conditioned_docking=true \
    protein_only=false \
    fix_pocket_seq=true \
    pocket_distance=${POCKET_DISTANCE} \
    skip_sampling=false \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des \
    denoiser_train_dir=${CKPT_DIR} \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=1 \
    sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=500 \
    sampling_cfg_overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=1 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=1 \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=5 \
    docking_metrics_cfg.pocket_distance_for_align=8.0 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"