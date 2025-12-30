#!/bin/bash
#SBATCH --job-name=redesign_pocket_seq
#SBATCH -p bioe,possu
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
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# pocket distances
POCKET_DISTANCES=(4.0 6.0 8.0)

# Experiment name
EXP_NAME=redesign_pocket_seq_replace_with_native_${POCKET_DISTANCES[${SLURM_ARRAY_TASK_ID}]}

# Input paths
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered_re/interface_df_full.parquet
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/caliby_wocenter_lmpnn_val_set_af2/samples

echo "Starting evaluation..."

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/redesign_pocket_seq.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    seed=0 \
    debug=false \
    evaluate_self_consistency=true \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    cached_example_path=${CACHED_EXAMPLE_PATH} \
    redesign_cfg.redesign_pocket_seq=true \
    redesign_cfg.use_native_pocket_seq=true \
    redesign_cfg.pocket_distance=${POCKET_DISTANCES[${SLURM_ARRAY_TASK_ID}]} \
    docking_metrics_cfg.pocket_distance_for_metrics=8.0 \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"