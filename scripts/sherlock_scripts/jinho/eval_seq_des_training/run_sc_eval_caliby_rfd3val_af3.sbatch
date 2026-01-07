#!/bin/bash
#SBATCH --job-name=run_sc_eval_rfd3_af3
#SBATCH -p possu
#SBATCH --array=0
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

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des

# Input paths
METADATA_PATH=null
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/caliby_rfd3val_af3/samples
PDB_NAME_LIST=null
PDB_NAME_EXT=.cif
LIGAND_SOURCE_PATH=/scratch/users/zhkim216/datasets/val_cifs/rfd3_ligand_conditional/lig_binder

# Exp name
EXP_NAME=caliby_rfd3val_af3

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
    ligand_source_cfg.add_ligands_to_designed_samples=true \
    ligand_source_cfg.source_is_designed=true \
    is_all_atom_sample=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for ${EXP_NAME}"