#!/bin/bash
#SBATCH --job-name=s3_a0
#SBATCH -p bioe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
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
EXP_NAME=s3_a0
GROUP=af3_monomer_scaffolding

# Dataset paths
PDB_PATH=/scratch/users/rshuai/datasets/af3_pdb_monomer
ANNOTATION_CSV=/scratch/users/rshuai/datasets/af3_pdb_monomer/pdb_key_annotations_for_eval2/annotated_pdb_keys.csv
CLUSTER_SAMPLE=true

# Run training script
WANDB__SERVICE_WAIT=300 python3 allatom_design/train_atom_denoiser.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    no_wandb=false \
    data.pdb_paths=[${PDB_PATH}] \
    data.annotation_csvs=[${ANNOTATION_CSV}] \
    data.cluster_sample=${CLUSTER_SAMPLE} \
    data.fixed_size=256 \
    data.subset_length_range=[32,256] \
    data.use_struct_preds=true \
    data.max_scrmsd=2.0 \
    data.max_rel_rog=1.4 \
    data.n_train_cluster_resample=10 \
    model.task=scaffold \
    scaffold_manager=scaffold_manager \
    denoiser.scaffold_module.enabled=false \
    denoiser.dit.scaffolding.use_concat=true \
    denoiser.dit.depth=15 \
    denoiser.dit.hidden_size=768 \
    train.batch_size=64 \
    train.compile_model=true \
    denoiser=dit_denoiser \
    trainer.check_val_every_n_epoch=4 \
    hydra.run.dir=${HYDRA_DIR} \
    out_dir=${OUT_DIR}
