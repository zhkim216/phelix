#!/bin/bash
#SBATCH --job-name=eval_seq_acc
#SBATCH -p bioe
#SBATCH --array=0-11
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%j.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
EXP_NAMES=("exp1_cfg0" "exp1_cfg1" "exp1_cfg2" "exp1_cfg3" "exp1_cfg4" "exp1_cfg6")
FIX_VALUES=("false" "true")
EXP_IDX=$((SLURM_ARRAY_TASK_ID//2))
FIX_IDX=$((SLURM_ARRAY_TASK_ID%2))
FIX_VALUE=${FIX_VALUES[FIX_IDX]}
EXP_NAME=eval_seq_acc_${EXP_NAMES[EXP_IDX]}_${FIX_VALUE}

# Inputs config
CKPT_PATH="/scratch/users/zhkim216/out_dir/ligand_seq_des/${EXP_NAMES[EXP_IDX]}/checkpoints"
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/interface_df_ver1.parquet

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=null \
    metadata_path=${METADATA_PATH} \
    sweep_cfg.enabled=true \
    sweep_cfg.ckpt_dir=${CKPT_PATH} \
    base_out_dir=/scratch/users/zhkim216/out_dir/eval_lc_seq_des \
    exp_name=${EXP_NAME} \
    input_cfg.load_from_cache=true \
    input_cfg.load_cache_cfg.cached_example_path=${CACHED_EXAMPLE_PATH} \
    sampling_cfg_overrides.batch_size=8 \
    sampling_cfg_overrides.num_seqs_per_pdb=8 \
    sampling_cfg_overrides.potts_sampling_cfg.potts_sweeps=500 \
    sampling_cfg_overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=${FIX_VALUE} \
    hydra.run.dir=${HYDRA_DIR}
