#!/bin/bash
#SBATCH --job-name=b4_aasd0_schedule
#SBATCH -p gpu_batch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-59%10
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/hielab/rshuai/slurm_log_files/job_%A_%a.err

# Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/hielab/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

# Load modules and activate conda environment
source ~/.rshuai_bash_profile
conda activate rshuai_allatom_design

# Set base directories
BASE_DIR=/home/talal/rshuai/allatom-design
HYDRA_DIR=/scratch/hielab/rshuai/hydra_outputs

# Model params
AF2_DATA_DIR=/scratch/hielab/rshuai/model_params/af2

# Dataset paths
PDB_PATH=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset
DESIGNABILITY_CSV=/scratch/hielab/rshuai/datasets/ingraham_cath_dataset/all_esmfold_native_with_cath_gapped_with_tm_with_seq.csv

# Set cache directories
export HF_HOME=/scratch/hielab/rshuai/cache/huggingface
export TORCH_HOME=/scratch/hielab/rshuai/cache/torch
OMEGAFOLD_CACHE=/scratch/hielab/rshuai/cache/omegafold/omegafold.pt

# Sweep parameters
num_steps_values=(1 10 20 100 200)
models=(
  b3_aasd5
  b4_aasd0_uniform_sqrt_t
  b4_aasd0_uniform_cbrt_t
  b4_aasd0_uniform_t
  b4_aasd0_uniform_cosine_t
  b4_aasd0_uniform_cubed_t
)
ckpts=(
  sd-step200000-epoch600.ckpt
  sd-step300000-epoch900.ckpt
)

# Calculate indices
num_models=${#models[@]}          # Should be 6
num_ckpts=${#ckpts[@]}            # Should be 2
num_num_steps=${#num_steps_values[@]}  # Should be 5

total_jobs=$(( num_models * num_ckpts * num_num_steps ))  # Should be 60

model_idx=$(( SLURM_ARRAY_TASK_ID / (num_ckpts * num_num_steps) ))
rem=$(( SLURM_ARRAY_TASK_ID % (num_ckpts * num_num_steps) ))
ckpt_idx=$(( rem / num_num_steps ))
num_steps_idx=$(( rem % num_num_steps ))

# Assign variables
MODEL_NAME=${models[$model_idx]}
CKPT=${ckpts[$ckpt_idx]}
NUM_STEPS=${num_steps_values[$num_steps_idx]}

# Denoiser checkpoint
SD_CKPT=/scratch/hielab/rshuai/allatom_design/allatom_design/${MODEL_NAME}/checkpoints/${CKPT}

# Wandb settings / experiment name
WANDB_ID=richardshuai
EXP_NAME=${MODEL_NAME}_sd_${CKPT}_S${NUM_STEPS}
GROUP=b4_aasd0_schedule

# Run inverse folding script
cd ${BASE_DIR}
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/inverse_fold.py \
    wandb_id=${WANDB_ID} \
    exp_name=${EXP_NAME} \
    group=${GROUP} \
    sd_ckpt=${SD_CKPT} \
    num_workers=${SLURM_CPUS_ON_NODE} \
    data.pdb_path=${PDB_PATH} \
    data.designability_csv=${DESIGNABILITY_CSV} \
    struct_pred_cfg.af2.data_dir=${AF2_DATA_DIR} \
    struct_pred_cfg.omegafold.cache_file=${OMEGAFOLD_CACHE} \
    temperature=0.0 \
    timestep_schedule.num_steps=${NUM_STEPS} \
    num_pdbs=300 \
    num_seqs_per_pdb=1 \
    batch_size=64 \
    hydra.run.dir=${HYDRA_DIR}
