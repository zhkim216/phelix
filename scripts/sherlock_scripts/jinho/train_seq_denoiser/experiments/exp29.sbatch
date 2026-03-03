#!/usr/bin/env bash
#SBATCH --job-name=exp29
#SBATCH -p possu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp29_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp29_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output/train_seq_denoiser "${HYDRA_DIR}" "${OUT_DIR}"

# Separate inductor cache per array task to avoid concurrent write corruption
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="training"
EXP_NAME_BASE="exp29"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data

# Path where processed pdb files are stored
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3"

# Experiment configs
DENOISER_TYPE="lc_atom_mpnn"

TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet"
VAL_METADATA_PATH="/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/metadata_lmpnnval_small_molecule_filtered_pocket_only.parquet"

CONFIGS=(        
  "data.pocket_featurizer_cfg.pocket_crop_distance=6.0"
  "data.pocket_featurizer_cfg.pocket_crop_distance=8.0"
  "data.pocket_featurizer_cfg.pocket_crop_distance=10.0"
  "data.pocket_featurizer_cfg.pocket_crop_distance=12.0"    
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
    data/dataset@data=ligandmpnn_pocket_only \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.val_metadata_path="${VAL_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.grouping_scheme="neighbor" \
    data.use_biologically_meaningful_small_molecule=true \
    data.exclude_small_molecules_covalently_linked_to_protein=true \
    data.pocket_only_training=true \
    data.pocket_featurizer_cfg.is_inference=false \
    data.pocket_featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.pocket_featurizer_cfg.remove_unresolved_tokens=true \
    data.pocket_featurizer_cfg.max_tokens=512 \
    data.pocket_featurizer_cfg.max_atoms=4608 \
    data.pocket_featurizer_cfg.spatial_crop_radius=15.0 \
    data.pocket_featurizer_cfg.training_structure_noise=0.1 \
    data.pocket_featurizer_cfg.asymmetric_noise=false \
    data.samples_per_epoch=50000 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.use_pocket_rbf=false \
    denoiser.mpnn.token_features.protein_graph_rbf_type="ncacocb" \
    denoiser.mpnn.token_features.use_sidechain_context=true \
    denoiser.mpnn.token_features.use_ligand_context=true \
    denoiser.mpnn.token_features.ligand_atom_context_num=25 \
    denoiser.mpnn.token_features.sidechain_context_token_num=16 \
    denoiser.mpnn.k_neighbors=32 \
    denoiser.mpnn.potts.k_neighbors_potts=32 \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"