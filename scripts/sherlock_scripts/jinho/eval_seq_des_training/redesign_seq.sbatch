#!/bin/bash
#SBATCH --job-name=redesign_pocket_seq
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
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Model names and denoiser train dirs
MODEL_NAMES=("exp2_cfg2" "exp2_cfg5")
MODEL_NAME=${MODEL_NAMES[$((SLURM_ARRAY_TASK_ID / 4))]}
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# pocket distances
POCKET_DISTANCES=(4.0 5.0 6.0 8.0)
POCKET_DISTANCE=${POCKET_DISTANCES[$((SLURM_ARRAY_TASK_ID % 4))]}

# Experiment name
EXP_NAME=caliby_wocenter_lmpnnval_redesign_${MODEL_NAME}_${POCKET_DISTANCE}

# Input paths
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered_re/interface_df_full.parquet
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/caliby_wocenter_lmpnn_val_set_af2/samples

echo "Starting evaluation..."

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/redesign_pocket_seq.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    debug=false \
    evaluate_self_consistency=true \
    start_step=null \
    end_step=92500 \
    eval_every_n_ckpts=4 \
    eval_last_ckpt=false \
    use_ema=true \
    metadata_path=${METADATA_PATH} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    cached_example_path=${CACHED_EXAMPLE_PATH} \
    redesign_cfg.redesign_pocket_seq=true \
    redesign_cfg.use_native_pocket_seq=false \
    redesign_cfg.pocket_distance=${POCKET_DISTANCE} \
    docking_metrics_cfg.pocket_distance_for_metrics=8.0 \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=1 \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"