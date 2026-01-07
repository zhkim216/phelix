#!/bin/bash
#SBATCH --job-name=run_sc_eval_rfd3_af3
#SBATCH -p possu,bioe
#SBATCH --array=0-4
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

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Model names
MODEL_NAMES=("pmpnn" "lmpnn005" "lmpnn010" "lmpnn020" "lmpnn030")
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Input paths
CACHED_EXAMPLE_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered/cached_examples
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_lmpnn_valset_filtered_re/interface_df_full.parquet

# Exp names and output paths
EXP_NAME=${MODEL_NAME}_rfd3val_af3
PDB_DIR=${BASE_OUT_DIR}/${EXP_NAME}/samples/backbones
PDB_NAME_LIST=null
PDB_NAME_EXT=.pdb

echo "Starting evaluation..."

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_sc_eval_af3.py \
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
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=${PDB_NAME_EXT} \
    add_ligands_to_designed_samples=false \
    is_all_atom_sample=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"