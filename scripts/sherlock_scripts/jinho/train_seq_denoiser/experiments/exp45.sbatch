#!/usr/bin/env bash
#SBATCH --job-name=exp45
#SBATCH -p possu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp45_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp45_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

# Separate inductor cache per array task to avoid concurrent write corruption
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p /scratch/users/zhkim216/job_output/train_seq_denoiser "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="training"
EXP_NAME_BASE="exp45"

# Dataset paths — v8 parquet with BMSM complete-linkage ligand clusters
# (q_pn_unit_bmsm_ligand_cluster_id) and physically-meaningful SM / metal columns.
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8"
TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8/metadata_ligval_seq_clustered_04_bmsm_ligclust.parquet"

# exp45: context-group chain-type whitelist + BMSM ligand-cluster + context_distance sweep.
# All cfgs enable the whitelist (PMSM / PMM / BRANCHED / DNA-RNA / peptide).
# cfg0: (i)   whitelist on, no tanimoto ligand cluster, d=5     (baseline vs. exp43_cfg0)
# cfg1: (ii)  whitelist on + BMSM ligand cluster,      d=5
# cfg2: (iii) whitelist on + BMSM ligand cluster,      d=6      (context distance sweep)
# cfg3: (iii) whitelist on + BMSM ligand cluster,      d=8      (context distance sweep)
DENOISER_TYPE="lc_atom_mpnn"

CONFIGS=(
  "data.sampling_weights.ligand_cluster_col=null                              data.context_distance=5.0"
  "data.sampling_weights.ligand_cluster_col=q_pn_unit_bmsm_ligand_cluster_id  data.context_distance=5.0"
  "data.sampling_weights.ligand_cluster_col=q_pn_unit_bmsm_ligand_cluster_id  data.context_distance=6.0"
  "data.sampling_weights.ligand_cluster_col=q_pn_unit_bmsm_ligand_cluster_id  data.context_distance=8.0"
)

IDX=$((SLURM_ARRAY_TASK_ID))
OVERRIDES="${CONFIGS[$IDX]}"
EXP_NAME="${EXP_NAME_BASE}_cfg$((SLURM_ARRAY_TASK_ID))"

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
    loss.main_seq_loss_pocket_only=false \
    loss.main_potts_loss_pocket_only=false \
    loss.seq_loss.per_token_avg=false \
    loss.potts.per_token_avg=false \
    data/dataset@data=ligandmpnn \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    +data.max_interface_distance=5.0 \
    data.exclude_val_cluster=true \
    data.grouping_scheme="neighbor" \
    data.pocket_only_training=false \
    data.interface_only_training=false \
    data.use_biologically_meaningful_small_molecule=true \
    data.exclude_small_molecules_covalently_linked_to_protein=true \
    data.samples_per_epoch=50000 \
    data.sampling_weights.k_percentile=90.0 \
    data.sampling_weights.alphas_interface.a_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.a_protein_nuc=0.0 \
    data.sampling_weights.alphas_interface.a_protein_peptide=0.0 \
    data.sampling_weights.alphas_interface.a_protein_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_protein_metal=0.0 \
    data.sampling_weights.alphas_interface.a_protein_loi=1.0 \
    data.recompute_context_columns=true \
    data.use_physically_meaningful_small_molecule=true \
    data.use_physically_meaningful_metal=true \
    data.context_group_chain_type_whitelist.enabled=true \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.training_structure_noise=0.1 \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=512 \
    data.featurizer_cfg.max_atoms=4608 \
    data.featurizer_cfg.crop_center_cutoff_distance=30.0 \
    data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25 \
    data.featurizer_cfg.pocket_distance=6.0 \
    data.featurizer_cfg.asymmetric_noise=false \
    data.featurizer_cfg.drop_prob_non_protein_chains=0.0 \
    denoiser.mpnn.k_neighbors=24 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.use_pocket_rbf=false \
    denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb \
    denoiser.mpnn.token_features.ligand_atom_context_num=36 \
    denoiser.mpnn.potts.k_neighbors_potts=24 \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    +mask_selector.pseudo_ligand_backbone_mask_radius=0 \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"
