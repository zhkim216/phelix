#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/mg_denovo/lc_seq_des_multi_mg_denovo_rasa_n100_mg_proto_gpt.sbatch
#SBATCH --job-name=lc_seq_des_mg_rasa_n100
#SBATCH -p possu,bioe
#SBATCH --array=0
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_mg_rasa_n100_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_mg_rasa_n100_%A_%a.err

set -euo pipefail

MODE="${MODE:-full}"  # full | smoke
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/eval_mg_denovo_rasa_n100_mg_proto_gpt
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
STAGED_DIR=/scratch/users/zhkim216/datasets/val_cifs/mg_denovo_val_cifs_test
DATASET_ID=mg_denovo_rasa_le0p25_uniform_bins_N100

MODEL_NAME=mg_proto_gpt
STEP=75000
VALSET_NAME=mg_denovo_rasa_n100
EXP_NAME=${DATASET_ID}_${MODEL_NAME}_ema0p99_no_guidance_af3
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

PDB_DIR=${STAGED_DIR}/cifs
SAMPLING_INPUTS_CSV=${STAGED_DIR}/sampling_inputs_${DATASET_ID}.csv

case "${MODE}" in
  smoke)
    PDB_NAME_LIST=${STAGED_DIR}/${DATASET_ID}_smoke2.txt
    NUM_CHUNKS=1
    CHUNK_IDX=0
    AF3_RECYCLES=1
    AF3_DIFFUSION_SAMPLES=1
    ;;
  full)
    PDB_NAME_LIST=${STAGED_DIR}/${DATASET_ID}.txt
    NUM_CHUNKS=1
    CHUNK_IDX=0
    AF3_RECYCLES=10
    AF3_DIFFUSION_SAMPLES=5
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected MODE=full or MODE=smoke." >&2
    exit 2
    ;;
esac

if [[ "${TASK_ID}" -ne 0 ]]; then
  echo "TASK_ID=${TASK_ID} is outside the single Mg N100 chunk; exiting without work."
  exit 0
fi

if [[ "${CHUNK_IDX}" -ge "${NUM_CHUNKS}" ]]; then
  echo "CHUNK_IDX=${CHUNK_IDX} is outside NUM_CHUNKS=${NUM_CHUNKS}; exiting without work."
  exit 0
fi

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

if [[ ! -d "${PDB_DIR}" ]]; then
  echo "Missing PDB_DIR=${PDB_DIR}. Run prepare_mg_denovo_rasa_n100_sherlock.py first." >&2
  exit 1
fi
if [[ ! -f "${PDB_NAME_LIST}" ]]; then
  echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}. Run prepare_mg_denovo_rasa_n100_sherlock.py first." >&2
  exit 1
fi
if [[ ! -f "${SAMPLING_INPUTS_CSV}" ]]; then
  echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}. Run prepare_mg_denovo_rasa_n100_sherlock.py first." >&2
  exit 1
fi

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des
NUM_SEQS_PER_PDB=2
BATCH_SIZE=1

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Starting Mg RASA N100 eval: mode=${MODE}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, step=${STEP}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name lc_seq_des_multi \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    num_workers=${CPUS_PER_TASK} \
    protein_only=false \
    run_seq_des=true \
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
    seq_des_cfg.atom_mpnn.overrides.use_potts_sampling=true \
    guidance_cfg.enabled=false \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    pocket_cfg.pocket_distance_for_docking_metrics=5.0 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.sample_lengths=null \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    cif_cfg.save.include_bonds=false \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=${AF3_RECYCLES} \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete: mode=${MODE}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, step=${STEP}"
