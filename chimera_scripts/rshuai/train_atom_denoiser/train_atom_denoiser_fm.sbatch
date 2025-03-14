#!/bin/bash
#SBATCH --job-name=u2_a1_lr1e-3
#SBATCH -p gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs
OUT_DIR=/scratch/hielab/rshuai/allatom_design
cd $BASE_DIR

# Set cache directories
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=u2_a1_lr1e-3
GROUP=af3_monomer

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/af3_pdb_monomer
ANNOTATION_CSV=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations_for_eval2/annotated_pdb_keys.csv
CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=8 \
    no_wandb=false \
    data.pdb_paths=[${PDB_PATH}] \
    data.annotation_csvs=[${ANNOTATION_CSV}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.fixed_size=256 \
    data.subset_length_range=[32,256] \
    data.use_struct_preds=true \
    data.max_scrmsd=1.0 \
    data.max_rel_rog=1.4 \
    data.n_train_cluster_resample=50 \
    data.translation_scale=0.0 \
    train.batch_size=64 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    denoiser/interpolant=sd3_rf_interpolant \
    optim.adam_inv_sqrt.ref_lr=1e-3 \
    trainer.check_val_every_n_epoch=2 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}