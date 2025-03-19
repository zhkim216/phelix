#!/bin/bash
#SBATCH --job-name=u2_a0
#SBATCH -p bioe
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design
WANDB_ID=richardshuai
cd ${BASE_DIR}

# Set cache directories
export TORCH_HOME=/scratch/users/rshuai/cache/torch

# Set experiment name and grouop
EXP_NAME=u2_a0
# GROUP=ai_cath
GROUP=af3_monomer

# Dataset paths
# PDB_PATH=/scratch/users/rshuai/datasets/ingraham_cath_dataset
# PDB_PATH=/scratch/users/subhang/datasets/augmented_ingraham_cath_bugfree
# ANNOTATION_CSV=null
# CLUSTER_SAMPLE=false
# USE_FIRST_SAMPLE=true

PDB_PATH=/scratch/users/rshuai/datasets/af3_pdb_monomer
ANNOTATION_CSV=/scratch/users/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations_for_eval2/annotated_pdb_keys.csv
CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 srun python3 allatom_design/train_atom_denoiser.py \
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
    data.max_scrmsd=2.0 \
    data.max_rel_rog=1.4 \
    data.n_train_cluster_resample=10 \
    data.use_struct_preds=true \
    train.batch_size=32 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    denoiser.dit.pair_rep.enabled=true \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}