#!/usr/bin/env bash
#SBATCH --job-name=exp8
#SBATCH -p possu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp8_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/exp8_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output/train_seq_denoiser "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="training"
EXP_NAME_BASE="exp8"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_lmpnn.parquet"

# Experiment configs
DENOISER_TYPE="lc_atom_mpnn"

# Path where processed pdb files are stored
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3"

CONFIGS=(  
  "denoiser.mpnn.token_features.protein_graph_rbf_type=ca denoiser.mpnn.ligand_conditioning=false data/dataset@data=ligandmpnn_ppionly"
  "denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb denoiser.mpnn.ligand_conditioning=false data/dataset@data=ligandmpnn_ppionly"
  "denoiser.mpnn.token_features.protein_graph_rbf_type=ca denoiser.mpnn.ligand_conditioning=true data/dataset@data=ligandmpnn"
  "denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb denoiser.mpnn.ligand_conditioning=true data/dataset@data=ligandmpnn"

  
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
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.training_structure_noise=0.3 \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=512 \
    data.featurizer_cfg.max_atoms=4608 \
    data.featurizer_cfg.crop_spatial_p=0.5 \
    data.featurizer_cfg.crop_center_cutoff_distance=15.0 \
    data.featurizer_cfg.pocket_distance=6.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_prot=1.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_nuc=0.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_peptide=0.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_small_molecule=0.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_metal=0.0 \
    data.sampling_weights.alphas_protein_monomer_chain.a_loi=0.0 \
    data.sampling_weights.alphas_interface.a_prot=1.0 \
    data.sampling_weights.alphas_interface.a_nuc=1.0 \
    data.sampling_weights.alphas_interface.a_peptide=1.0 \
    data.sampling_weights.alphas_interface.a_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_metal=1.0 \
    data.sampling_weights.alphas_interface.a_loi=1.0 \
    data.sampling_weights.betas.beta_protein_monomer_chain=1.0 \
    data.sampling_weights.betas.beta_interface=1.0 \
    data.samples_per_epoch=50000 \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"