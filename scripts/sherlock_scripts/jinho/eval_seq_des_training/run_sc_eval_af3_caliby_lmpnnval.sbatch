#!/bin/bash
#SBATCH --job-name=sc_eval_caliby_lmpnnval
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/sc_eval_caliby_lmpnnval_%A.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/sc_eval_caliby_lmpnnval_%A.err

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

# Input paths (caliby-designed lmpnn val CIFs, flat directory, 147 files)
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/caliby_samples/lmpnn_val_cifs/samples
VALSET_NAME="lmpnnval"

# Experiment name
EXP_NAME=eval_caliby_${VALSET_NAME}_af3

echo "Starting SC eval for valset=${VALSET_NAME}"

# Run evaluation script (no array job needed for 147 files)
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_sc_eval_af3.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    protein_only=false \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=null \
    pdb_cfg.pdb_name_ext=".cif" \
    pocket_cfg.pocket_distance_for_metrics=6.0 \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "SC eval complete for valset=${VALSET_NAME}"
