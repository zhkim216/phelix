#!/bin/bash
#SBATCH --job-name=lc_seq_des_denovoval_e37c11_s22500_l350_450
#SBATCH -p possu,bioe
#SBATCH --array=0-255%8
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_exp37_cfg11_step22500_len350_450_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_multi_denovoval_exp37_cfg11_step22500_len350_450_%A_%a.err

# ==============================================================================
# Dataset: denovo_val_cifs (flat structure)
# Model/step: exp37_cfg11 @ step 22500
# Lengths: [350,450]
# num_seqs_per_pdb=2, batch_size=1
#
# Array mapping:
#   NUM_BUCKETS=4 (indices bucketed as 0-4, 5-9, 10-14, 15-19)
#   NUM_CHUNKS=64
#   TOTAL_TASKS=4 x 64 = 256  -> --array=0-255%8
# ==============================================================================

MODEL_NAME="exp37_cfg11"
STEP=22500
NUM_BUCKETS=4
NUM_CHUNKS=64

BUCKET_IDX=$((SLURM_ARRAY_TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((SLURM_ARRAY_TASK_ID % NUM_CHUNKS))

SAMPLE_INDICES_BY_BUCKET=(
    "[0,1,2,3,4]"
    "[5,6,7,8,9]"
    "[10,11,12,13,14]"
    "[15,16,17,18,19]"
)
SAMPLE_INDICES=${SAMPLE_INDICES_BY_BUCKET[$BUCKET_IDX]}

if [[ -z "${SAMPLE_INDICES}" ]]; then
    echo "Invalid BUCKET_IDX=${BUCKET_IDX} for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
JOB_OUT_DIR=/scratch/users/zhkim216/job_output/eval_ligand_seq_des

mkdir -p "${JOB_OUT_DIR}" "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

# Input paths (denovo val CIFs, flat directory)
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
VALSET_NAME="denovoval"

# Eval settings
BATCH_SIZE=1
NUM_SEQS_PER_PDB=2

# Denoiser train directory
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

# Experiment name
EXP_NAME=eval_${MODEL_NAME}_${VALSET_NAME}_seq${NUM_SEQS_PER_PDB}_af3_len350_450

echo "Starting evaluation: model=${MODEL_NAME}, step=${STEP}, lengths=[350,450], indices=${SAMPLE_INDICES}, bucket=${BUCKET_IDX}/${NUM_BUCKETS}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, array_task=${SLURM_ARRAY_TASK_ID}"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "${COMPUTE_CAP}" | cut -d. -f1)

if [[ "${MAJOR}" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

# Run evaluation script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name lc_seq_des_multi \
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
    seq_des_cfg.atom_mpnn.overrides.batch_size=${BATCH_SIZE} \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
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
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    "pdb_cfg.sample_indices=${SAMPLE_INDICES}" \
    'pdb_cfg.sample_lengths=[350,450]' \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete: model=${MODEL_NAME}, step=${STEP}, bucket=${BUCKET_IDX}, chunk=${CHUNK_IDX}"
