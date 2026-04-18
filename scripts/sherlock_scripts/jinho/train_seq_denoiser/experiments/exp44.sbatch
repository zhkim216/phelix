#!/usr/bin/env bash
#SBATCH --job-name=exp44
#SBATCH -p possu
<<<<<<< HEAD
#SBATCH --array=0
=======
#SBATCH --array=1-5
>>>>>>> refs/remotes/origin/jinho/AAA
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp44_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp44_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

# Separate inductor cache per array task to avoid concurrent write corruption
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_exp44_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

mkdir -p /scratch/users/zhkim216/job_output/train_seq_denoiser "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="training"
EXP_NAME_BASE="exp44"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3"
TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet"

# cfg0: Same as exp43_cfg0 (enc-dec + seq_loss), 
# cfg1: Potts composite off; pseudolikelihood NLL + seq_loss on
# cfg2: Potts pseudolikelihood only (composite + seq_loss both off)
# cfg3: Potts parameterization factor -> score
# cfg4: Potts parameterization factor -> factor_scale
# cfg5: Potts parameterization factor -> score_scale
DENOISER_TYPE="lc_atom_mpnn"

# Per-cfg overrides indexed by SLURM_ARRAY_TASK_ID (cfg0 slot is a placeholder;
# array=1-5 means only ${CONFIGS[1]}..${CONFIGS[5]} are ever accessed).
CONFIGS=(
  ""  # cfg0 placeholder (identical to exp43_cfg0; not re-trained here)
  "loss.loss_weights.potts_composite_loss=0.0 loss.loss_weights.potts_pseudolikelihood_loss=1.0"
  "loss.loss_weights.potts_composite_loss=0.0 loss.loss_weights.potts_pseudolikelihood_loss=1.0 loss.loss_weights.seq_loss=0.0"
  "denoiser.mpnn.potts.parameterization=score"
  "denoiser.mpnn.potts.parameterization=factor_scale"
  "denoiser.mpnn.potts.parameterization=score_scale"
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
    train.batch_size=16 \
    trainer.accumulate_grad_batches=2 \
    train.debug=false \
    train.compile.compile_model=true \
    trainer.strategy="ddp_find_unused_parameters_true" \
    trainer.enable_progress_bar=true \
    resume.use_current_cfg=true \
    loss.main_seq_loss_pocket_only=false \
    loss.main_potts_loss_pocket_only=false \
    loss.seq_loss.per_token_avg=false \
    loss.potts.per_token_avg=false \
    data/dataset@data=ligandmpnn \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.sampling_weights.ligand_cluster_col=null \
    +data.max_interface_distance=5.0 \
    data.exclude_val_cluster=true \
    data.grouping_scheme="neighbor" \
    data.pocket_only_training=false \
    data.interface_only_training=false \
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
    data.sampling_weights.alphas_interface.a_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.a_protein_nuc=0.0 \
    data.sampling_weights.alphas_interface.a_protein_peptide=0.0 \
    data.sampling_weights.alphas_interface.a_protein_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_protein_metal=0.0 \
    data.sampling_weights.alphas_interface.a_protein_loi=1.0 \
    data.use_biologically_meaningful_small_molecule=true \
    data.exclude_small_molecules_covalently_linked_to_protein=true \
    data.samples_per_epoch=50000 \
    data.sampling_weights.k_percentile=90.0 \
    denoiser.mpnn.k_neighbors=24 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.use_pocket_rbf=false \
    denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb \
    denoiser.mpnn.token_features.ligand_atom_context_num=36 \
    denoiser.mpnn.potts.k_neighbors_potts=24 \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    mask_selector.pseudo_ligand_backbone_mask_radius=0 \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"
