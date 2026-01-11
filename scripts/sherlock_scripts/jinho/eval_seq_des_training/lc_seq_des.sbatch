#!/bin/bash
#SBATCH --job-name=lc_seq_des
#SBATCH -p bioe,possu
#SBATCH --array=0-17
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Model names (2 models) x Steps (9 steps) = 18 combinations
# Array index mapping: model_idx = task_id / 9, step_idx = task_id % 9
MODEL_NAMES=("exp2_cfg2" "exp2_cfg5")
STEPS=(2500 12500 22500 32500 42500 52500 62500 72500 82500)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 9))
STEP_IDX=$((SLURM_ARRAY_TASK_ID % 9))

MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
STEP=${STEPS[$STEP_IDX]}
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name 
EXP_NAME=eval_${MODEL_NAME}_rfd3val_af3

# Input paths
METADATA_PATH=null
CACHED_EXAMPLE_PATH=null
POS_CONSTRAINT_CSV=null
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/rfd3_ligand_conditional/lig_binder
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/rfd3_ligand_conditional/rfd3_min_max_median_len150_filtered_nonsimilar_ccd_clustered03.txt
LIGAND_SOURCE_PATH=/scratch/users/zhkim216/datasets/val_cifs/rfd3_ligand_conditional/lig_binder
ADD_LIGANDS_TO_DESIGNED_SAMPLES=false
SOURCE_IS_DESIGNED=true

echo "Starting evaluation for model=${MODEL_NAME} (idx=${MODEL_IDX}), step=${STEP} (idx=${STEP_IDX})"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=baseline \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    metadata_path=${METADATA_PATH} \
    pos_constraint_csv=${POS_CONSTRAINT_CSV} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.overrides.batch_size=8 \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=1 \
    redesign_cfg.use_native_seq=false \
    docking_metrics_cfg.pocket_distance_for_metrics=8.0 \
    is_all_atom_sample=false \
    sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    ligand_source_cfg.ligand_source_path=${LIGAND_SOURCE_PATH} \
    ligand_source_cfg.add_ligands_to_designed_samples=${ADD_LIGANDS_TO_DESIGNED_SAMPLES} \
    ligand_source_cfg.source_is_designed=${SOURCE_IS_DESIGNED} \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=triton \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}"
