#!/bin/bash
#SBATCH --job-name=augmented_af3_monomer_v2_fampnn_multi
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-199%10
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
ml python
module load gcc/12.4.0
module load cuda/12.1
ENV_DIR=/oak/stanford/groups/possu/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
OUT_DIR=/scratch/users/rshuai/allatom_design
cd ${BASE_DIR}

# Inputs config
PDB_DIR=/scratch/users/rshuai/datasets/af3_pdb_monomer/preprocessing/residx_quality_control_af3_monomer/filtered_mmcifs

# Set HF transformers cache
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# FAMPNN checkpoint path
CHECKPOINT_PATH=/scratch/users/rshuai/allatom_design/allatom_design/c6_aasd1/checkpoints/ema/sd-step120000-epoch344-ema0.99.ckpt

# Output directory
OUT_DIR=/scratch/users/rshuai/datasets/fampnn_preds/augmented_af3_monomer_v2/S20_nl0.1_n4_L257_512

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/fampnn_multi.py \
    input_cfg.pdb_dir=${PDB_DIR} \
    input_cfg.subset_length_range=[257,512] \
    input_cfg.presort_by_length=true \
    input_cfg.array_id=${SLURM_ARRAY_TASK_ID} \
    input_cfg.num_arrays=200 \
    checkpoint_path=${CHECKPOINT_PATH} \
    out_dir=${OUT_DIR} \
    run_self_consistency_eval=true \
    sampling_cfg_overrides.batch_size=32 \
    sampling_cfg_overrides.num_seqs_per_pdb=4 \
    sampling_cfg_overrides.num_steps=20 \
    sampling_cfg_overrides.noise_labels=0.1 \
    sampling_cfg_overrides.add_noise=false \
    hydra.run.dir=${HYDRA_DIR} \
