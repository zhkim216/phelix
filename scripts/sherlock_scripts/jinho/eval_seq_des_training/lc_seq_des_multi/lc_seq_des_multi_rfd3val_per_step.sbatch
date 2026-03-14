#!/bin/bash
#SBATCH --job-name=lc_seq_des_rfd3val_old_per_step
#SBATCH -p bioe,owners
#SBATCH --array=0-899  # Update this: 3 models x NUM_STEPS x 50 folders - 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_rfd3val_old_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_rfd3val_old_%A_%a.err

# ==============================================================================
# USER CONFIG: Modify these arrays as needed
# ==============================================================================
MODEL_NAMES=("exp18_cfg3" "exp36_cfg0" "exp36_cfg1")
STEPS=(2500 12500 22500 32500 42500 52500)  # Modify this array with desired steps
NUM_CCD_FOLDERS=50

# ==============================================================================
# Array index calculation
# Total tasks = 3 models x NUM_STEPS x 50 folders
# Update --array above when changing STEPS!
# Example: STEPS=(2500 22500 42500) -> 3 x 3 x 50 = 450 -> --array=0-449
# ==============================================================================
NUM_MODELS=${#MODEL_NAMES[@]}
NUM_STEPS=${#STEPS[@]}
TASKS_PER_MODEL=$((NUM_STEPS * NUM_CCD_FOLDERS))

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_MODEL))
REMAINDER=$((SLURM_ARRAY_TASK_ID % TASKS_PER_MODEL))
STEP_IDX=$((REMAINDER / NUM_CCD_FOLDERS))
FOLDER_IDX=$((REMAINDER % NUM_CCD_FOLDERS))

MODEL_NAME=${MODEL_NAMES[$MODEL_IDX]}
STEP=${STEPS[$STEP_IDX]}

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
PROJECT=eval_ligand_seq_des

# Input paths (rfd3 val CIFs with CCD code subdirectories)
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/rfd3_val_cifs/samples_uncompressed
VALSET_NAME="rfd3val"

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3

echo "Starting evaluation for model=${MODEL_NAME}, step=${STEP}, valset=${VALSET_NAME}, folder=${FOLDER_IDX}/50 (array_task=${SLURM_ARRAY_TASK_ID})"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    protein_only=false \
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
    sampling_inputs_csv=null \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    pocket_cfg.pocket_distance_for_metrics=6.0 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=null \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=true \
    pdb_cfg.split_by_subfolder=true \
    pdb_cfg.array_id=${FOLDER_IDX} \
    pdb_cfg.num_arrays=50 \
    'pdb_cfg.sample_indices=[0,1,2,3,4]' \
    'pdb_cfg.sample_lengths=[150,250]' \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, step=${STEP}, folder=${FOLDER_IDX}"
