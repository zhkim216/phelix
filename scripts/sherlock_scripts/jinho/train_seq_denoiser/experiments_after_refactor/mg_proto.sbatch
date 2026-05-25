#!/usr/bin/env bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/train_seq_denoiser/experiments_after_refactor/mg_proto.sbatch
#SBATCH --job-name=mg_proto
#SBATCH -p possu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/mg_proto_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/mg_proto_%A_%a.err

set -euo pipefail

BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/train_seq_denoiser/mg_proto
OUT_DIR=/scratch/users/zhkim216/out_dir
JOB_OUTPUT_DIR=/scratch/users/zhkim216/job_output/train_seq_denoiser

mkdir -p "${JOB_OUTPUT_DIR}" "${HYDRA_DIR}" "${OUT_DIR}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_mg_proto_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP=training

# External evidence policy / exp name
POLICIES=(no_filter substring gpt)
IDX="${ARRAY_TASK_ID}"
if (( IDX < 0 || IDX >= ${#POLICIES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0..$((${#POLICIES[@]} - 1))" >&2
  exit 2
fi

POLICY="${POLICIES[$IDX]}"
EXP_NAME="mg_proto_${POLICY}"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9"
TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9/metadata_ligval_seq_clustered_03_with_mg_pubmed_evidence.parquet"
VALIDATION_IDS_FILE="/scratch/users/zhkim216/datasets/val_cifs/mg_native_val_cifs_test/mg_nativeval_sm_all.txt"
VAL_METADATA_PATH="/scratch/users/zhkim216/datasets/val_cifs/mg_native_val_cifs_test/metadata_for_training_mg_nativeval_sm_all.parquet"

# Denoiser type
DENOISER_TYPE="lc_atom_mpnn"

echo "Starting ${EXP_NAME}"
echo "  policy: ${POLICY}"
echo "  base_dir: ${BASE_DIR}"
echo "  out_dir: ${OUT_DIR}"
echo "  hydra_dir: ${HYDRA_DIR}/${EXP_NAME}"
echo "  torchinductor_cache: ${TORCHINDUCTOR_CACHE_DIR}"

DEBUG= HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 \
python3 allatom_design/train_seq_denoiser.py \
    --config-path configs/seq_denoiser \
    --config-name mg_proto.yaml \
    wandb.wandb_id="${WANDB_ID}" \
    wandb.project="${PROJECT}" \
    wandb.group="${GROUP}" \
    wandb.no_wandb=false \
    num_workers="${CPUS_PER_TASK}" \
    out_dir="${OUT_DIR}" \
    exp_name="${EXP_NAME}" \
    train.debug=false \
    train.batch_size=32 \
    train.compile.compile_model=true \
    trainer.accumulate_grad_batches=2 \
    trainer.strategy="ddp_find_unused_parameters_true" \
    denoiser.name="${DENOISER_TYPE}" \
    resume.use_current_cfg=true \
    loss.main_seq_loss_pocket_only=false \
    loss.main_potts_loss_pocket_only=false \
    loss.seq_loss.per_token_avg=false \
    loss.potts.per_token_avg=false \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.validation_ids_file="${VALIDATION_IDS_FILE}" \
    data.val_metadata_path="${VAL_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.interface_only_training=false \
    data.samples_per_epoch=50000 \
    data.mg_proto.external_evidence_policy="${POLICY}" \
    data.sampling_weights.alphas_interface.alpha_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_small_molecule=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_metal=1.0 \
    data.sampling_weights.alphas_interface.alpha_protein_nuc=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_nuc_lig=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_peptide=0.0 \
    data.sampling_weights.k_percentile=90.0 \
    data.sampling_weights.ligand_cluster_col=null \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.training_structure_noise=0.1 \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=256 \
    data.featurizer_cfg.max_atoms=2304 \
    data.featurizer_cfg.crop_center_cutoff_distance=15.0 \
    data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25 \
    data.featurizer_cfg.drop_prob_non_protein_chains=0.0 \
    data.featurizer_cfg.pocket_distance=5.0 \
    data.featurizer_cfg.pocket_n_min_ligand_atoms=1 \
    data.featurizer_cfg.use_pocket_rbf=false \
    data.featurizer_cfg.asymmetric_noise=false \
    denoiser.mpnn.k_neighbors=24 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.use_pocket_rbf=false \
    denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb \
    denoiser.mpnn.token_features.ligand_atom_context_num=25 \
    denoiser.mpnn.potts.k_neighbors_potts=24 \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    hydra.run.dir="${HYDRA_DIR}/${EXP_NAME}"
