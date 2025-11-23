#!/usr/bin/env bash
#SBATCH --job-name=lp_exp1
#SBATCH -p possu
#SBATCH --array=6
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A_%a.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Wandb settings / experiment name
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP="atomworks"
EXP_NAME_BASE="exp1"

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data

# Experiment configs
DENOISER_TYPE="lc_atom_mpnn"

CONFIGS=(
  # "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_lmpnn_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=1.0"
  # "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_lmpnn_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=3.0"
  # "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_lmpnn_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=5.0"
  # "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_af3_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=1.0"
  # "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_af3_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=3.0"
  "data.pdb_path=/scratch/users/zhkim216/datasets/atomworks_af3_re data.sampling_weights.betas.beta_chain=1.0 data.sampling_weights.betas.beta_interface=5.0"
)

IDX=$((SLURM_ARRAY_TASK_ID) -6)
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
    data.validation_ids_file="/scratch/users/zhkim216/datasets/splits/lmpnn_val_list.csv" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.data_loader_cfg.prefetch_factor=1 \
    data.data_loader_cfg.prefetch_buffer_size=1 \
    data.data_loader_cfg.train_pin_memory=true \
    data.data_loader_cfg.train_persistent_workers=true \
    data.data_loader_cfg.val_pin_memory=false \
    data.data_loader_cfg.val_persistent_workers=false \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=512 \
    data.featurizer_cfg.max_atoms=4608 \
    data.sampling_weights.alphas_chain.a_prot=1.0 \
    data.sampling_weights.alphas_chain.a_small_molecule=0.0 \
    data.sampling_weights.alphas_chain.a_nuc=0.0 \
    data.sampling_weights.alphas_chain.a_metal=0.0 \
    data.sampling_weights.alphas_interface.a_prot=1.0 \
    data.sampling_weights.alphas_interface.a_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_nuc=1.0 \
    data.sampling_weights.alphas_interface.a_metal=1.0 \
    data.sampling_weights.betas.beta_chain=1.0 \
    data.sampling_weights.betas.beta_interface=1.0 \
    data.samples_per_epoch=50000 \
    mask_selector.restype_masking_schedule=uniform_t \
    mask_selector.atom_masking_schedule=uniform_t \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"