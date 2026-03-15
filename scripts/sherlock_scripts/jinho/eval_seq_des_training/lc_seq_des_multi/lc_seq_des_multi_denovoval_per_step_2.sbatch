#!/bin/bash
#SBATCH --job-name=lc_seq_des_denovoval_2
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-999
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_%A_%a.err

# ==============================================================================
# Dataset: denovo_val_cifs (flat structure, 64 CCD codes)
# Files: {CCD}_len_{L}_{IDX}_model_{M}.cif.gz
# Lengths filtered to 150, 250 only; sample_indices [0,1,2,3,4]
#
# Total: 15 models x 8 steps x 64 chunks = 7680 tasks
# This is part 2/7 (global 2000-2999)
# ==============================================================================
MODEL_NAMES=("exp37_cfg0" "exp37_cfg1" "exp37_cfg2" "exp37_cfg3" "exp37_cfg4" "exp37_cfg5"
             "exp18_cfg3" "exp36_cfg0" "exp36_cfg1" "exp37_cfg6" "exp37_cfg7" "exp37_cfg8"
             "exp37_cfg9" "exp37_cfg10" "exp37_cfg11")
STEPS=(2500 12500 22500 32500 42500 52500 62500 72500)
NUM_CHUNKS=64

# Global offset for this part
GLOBAL_OFFSET=2000
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + GLOBAL_OFFSET))

# ==============================================================================
# Array index calculation
# ==============================================================================
NUM_MODELS=${#MODEL_NAMES[@]}
NUM_STEPS=${#STEPS[@]}
TASKS_PER_MODEL=$((NUM_STEPS * NUM_CHUNKS))

MODEL_IDX=$((GLOBAL_TASK_ID / TASKS_PER_MODEL))
REMAINDER=$((GLOBAL_TASK_ID % TASKS_PER_MODEL))
STEP_IDX=$((REMAINDER / NUM_CHUNKS))
CHUNK_IDX=$((REMAINDER % NUM_CHUNKS))

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

# Input paths (denovo val CIFs, flat directory)
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
VALSET_NAME="denovoval"

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_af3

echo "Starting evaluation for model=${MODEL_NAME}, step=${STEP}, valset=${VALSET_NAME}, chunk=${CHUNK_IDX}/${NUM_CHUNKS} (array_task=${SLURM_ARRAY_TASK_ID}, global=${GLOBAL_TASK_ID})"

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
    seq_des_cfg.atom_mpnn.overrides.batch_size=2 \
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
    pdb_cfg.pdb_name_ext=".cif.gz" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    'pdb_cfg.sample_indices=[0,1,2,3,4]' \
    'pdb_cfg.sample_lengths=[150,250]' \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete for model=${MODEL_NAME}, step=${STEP}, chunk=${CHUNK_IDX}"
