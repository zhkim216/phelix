#!/usr/bin/env bash
#SBATCH --job-name=exp19
#SBATCH -p possu
#SBATCH --array=0-6
#SBATCH --nodelist=sh04-10n01
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp18_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp18_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

# Separate inductor cache per array task to avoid concurrent write corruption
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p /scratch/users/zhkim216/job_output/train_seq_denoiser "${HYDRA_DIR}" "${OUT_DIR}"

# Separate inductor cache per array task to avoid concurrent write corruption
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="training"
EXP_NAME_BASE="exp19"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data

# Path where processed pdb files are stored
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3"

# Experiment configs
DENOISER_TYPE="lc_atom_mpnn"

TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet"

CONFIGS=(      
  "denoiser.mpnn.token_features.use_pocket_rbf=false"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncaco denoiser.mpnn.token_features.pocket_rbf_distance=4.0"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncacocb denoiser.mpnn.token_features.pocket_rbf_distance=4.0"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncaco denoiser.mpnn.token_features.pocket_rbf_distance=4.5"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncacocb denoiser.mpnn.token_features.pocket_rbf_distance=4.5"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncaco denoiser.mpnn.token_features.pocket_rbf_distance=5.0"
  "denoiser.mpnn.token_features.use_pocket_rbf=true denoiser.mpnn.token_features.pocket_rbf_type=ncacocb denoiser.mpnn.token_features.pocket_rbf_distance=5.0"
)

IDX=$((SLURM_ARRAY_TASK_ID))
OVERRIDES="${CONFIGS[$IDX]}"
EXP_NAME="${EXP_NAME_BASE}_cfg${SLURM_ARRAY_TASK_ID}"

# Run model
HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 \
python3 allatom_design/train_seq_denoiser.py \
    --config-dir allatom_design/configs/seq_denoiser \
    --config-name seq_denoiser.yaml \
    wandb.wandb_id="${WANDB_ID}" \
    wandb.project="${PROJECT}" \
    wandb.group="${GROUP}" \
    wandb.no_wandb=false \
    num_workers=${SLURM_CPUS_PER_TASK} \
    out_dir="${OUT_DIR}" \
    exp_name="${EXP_NAME}" \
    denoiser="${DENOISER_TYPE}" \
    train.batch_size=8 \
    trainer.accumulate_grad_batches=4 \
    train.debug=false \
    train.compile.compile_model=true \
    trainer.strategy="ddp_find_unused_parameters_true" \
    resume.use_current_cfg=true \
    data/dataset@data=ligandmpnn \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.grouping_scheme="neighbor" \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.training_structure_noise=0.3 \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=512 \
    data.featurizer_cfg.max_atoms=4608 \
    data.sampling_weights.alphas_interface.a_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.a_protein_nuc=0.0 \
    data.sampling_weights.alphas_interface.a_protein_peptide=0.0 \
    data.sampling_weights.alphas_interface.a_protein_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_protein_metal=0.0 \
    data.sampling_weights.alphas_interface.a_protein_loi=1.0  \
    data.featurizer_cfg.crop_center_cutoff_distance=15.0 \
    data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25 \
    data.featurizer_cfg.pocket_distance=6.0 \
    data.use_biologically_meaningful_small_molecule=true \
    data.exclude_small_molecules_covalently_linked_to_protein=true \
    data.samples_per_epoch=50000 \
    data.sampling_weights.k_percentile=90.0 \
    data.featurizer_cfg.drop_prob_non_protein_chains=0.0 \
    denoiser.mpnn.token_features.protein_graph_rbf_type=ca \
    denoiser.mpnn.token_features.ligand_atom_context_num=25 \
    denoiser.mpnn.ligand_conditioning=true \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"