#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%j.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
OUT_DIR=/scratch/users/zhkim216/out_dir

mkdir -p /scratch/users/zhkim216/job_output "${HYDRA_DIR}" "${OUT_DIR}"

cd "${BASE_DIR}"

# Dataset paths
PDB_PATH=/scratch/users/zhkim216/datasets/atomworks
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks/cached_residue_data

# Experiment configs
DENOISER_TYPE="debug_lc_atom_mpnn"

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_seq_denoiser.py \
    --config-dir allatom_design/configs/seq_denoiser \
    --config-name debug_dummy_seq_denoiser.yaml \
    wandb.wandb_id="${WANDB_ID}" \
    wandb.project="${PROJECT}" \
    wandb.group="${GROUP}" \
    wandb.no_wandb=false \
    num_workers="${SLURM_CPUS_PER_TASK}" \
    out_dir="${OUT_DIR}" \
    exp_name="${EXP_NAME}" \
    denoiser="${DENOISER_TYPE}" \
    train.batch_size=64 \
    trainer.accumulate_grad_batches=1 \
    train.debug=false \
    data.pdb_path="${PDB_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.featurizer_cfg.max_tokens=256 \
    data.featurizer_cfg.max_atoms=2304 \
    data.sampling_weights.alphas_chain.a_prot=3.0 \
    data.sampling_weights.alphas_chain.a_peptide=0.0 \
    data.sampling_weights.alphas_chain.a_nuc=0.0 \
    data.sampling_weights.alphas_chain.a_small_molecule=0.0 \
    data.sampling_weights.alphas_chain.a_metal=0.0 \
    data.sampling_weights.alphas_chain.a_loi=0.0 \
    data.sampling_weights.alphas_interface.a_prot=1.0 \
    data.sampling_weights.alphas_interface.a_peptide=0.0 \
    data.sampling_weights.alphas_interface.a_nuc=0.0 \
    data.sampling_weights.alphas_interface.a_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.a_metal=0.0 \
    data.sampling_weights.alphas_interface.a_loi=0.0 \
    data.sampling_weights.betas.beta_chain=1.0 \
    data.sampling_weights.betas.beta_interface=1.0 \
    data.samples_per_epoch=50000 \
    mask_selector.restype_masking_schedule=uniform_t \
    mask_selector.atom_masking_schedule=uniform_t \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}"