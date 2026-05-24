#!/bin/bash
#SBATCH --job-name=sc_lmpnn010_denovoval_designable_pd6
#SBATCH -p possu,bioe,owners
#SBATCH --array=0-60
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/sc_lmpnn010_denovoval_designable_pd6_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/sc_lmpnn010_denovoval_designable_pd6_%A_%a.err

# ==============================================================================
# Self-consistency evaluation for LigandMPNN 0.10 designed samples on denovo val
# 61 chunks, each chunk processes a subset of designed backbones
# ==============================================================================
NUM_CHUNKS=61
CHUNK_IDX=${SLURM_ARRAY_TASK_ID}

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Input paths
PDB_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_lmpnn010_denovoval_designable_pd6_af3/samples/backbones
EXP_NAME=eval_lmpnn010_denovoval_designable_pd6_af3

echo "Starting SC evaluation for lmpnn010 denovoval designable pd6, chunk=${CHUNK_IDX}/${NUM_CHUNKS} (array_task=${SLURM_ARRAY_TASK_ID})"

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
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/run_sc_eval_af3.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval \
    --config-name run_sc_eval_af3 \
    wandb.no_wandb=true \
    debug=false \
    input_sample_is_designed=true \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    pdb_cfg.sample_indices=null \
    pdb_cfg.sample_lengths=null \
    pocket_cfg.pocket_distance_for_docking_metrics=8.0 \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "SC evaluation complete for lmpnn010 denovoval designable pd6, chunk=${CHUNK_IDX}"
