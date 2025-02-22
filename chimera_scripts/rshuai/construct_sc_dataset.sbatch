#!/bin/bash
#SBATCH --job-name=construct_sc_dataset
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err
#SBATCH --array=10-49%2

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch

# Set paths
NOISE_LABELS=0.0
MODEL_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/post_hoc_ema_ckpts/ema-step300000-std0.220.ckpt
PDB_DIR=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/train_mmcifs
PDB_KEY_LIST=/scratch/hielab/rshuai/datasets/af3_pdb_monomer/train_pdb_keys.list
OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/af3_pdb_monomer_L32_256_nl${NOISE_LABELS}_eval

# PDB_KEY_LIST=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/eval_pdb_keys.list
# OUT_DIR=/scratch/hielab/rshuai/allatom_design/allatom_design/c2_aasd2/construct_sc_dataset/ingraham_cath_L151_256_S50_val

cd ${BASE_DIR}

# Run the Python script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/construct_sc_dataset.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    hydra.run.dir=${HYDRA_DIR} \
    checkpoint_path=${MODEL_CKPT} \
    pdb_dir=${PDB_DIR} \
    pdb_key_list=${PDB_KEY_LIST} \
    pdb_key_ext=.cif \
    subset_length_range=[32,256] \
    presort_by_length=true \
    out_dir=${OUT_DIR} \
    num_steps=50 \
    noise_labels=${NOISE_LABELS} \
    temperature_list=[0.1,0.5,1.0] \
    num_seqs_per_pdb_list=[1,1,2] \
    array_id=${SLURM_ARRAY_TASK_ID} \
    num_arrays=100
