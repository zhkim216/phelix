#!/bin/bash
#SBATCH --job-name=a7_cbb14_eval_sweep_a0
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --array=0-143%8                    # 144 tasks with max 8 concurrent jobs
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%A_%a.err
#

## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh

conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/code/allatom-design
HYDRA_DIR=/scratch/users/rshuai/hydra_outputs
MPNN_PARAMS_DIR=/scratch/users/rshuai/model_params/mpnn

cd ${BASE_DIR}

# Dataset paths
NNTM_DATASET=/scratch/users/rshuai/datasets/ingraham_cath_dataset/pdb_store_renumbered

# Set HF transformers cache
export HF_HOME=/scratch/users/rshuai/cache/huggingface

# Denoiser checkpoint
AD_CKPT=/scratch/users/rshuai/allatom_design/allatom_design/a7_cbb14/checkpoints/ad-epoch6859.ckpt

# Define parameter grids
ca_modes=("linear" "square" "cubic" "sqrt")
nco_modes=("linear" "square" "cubic" "sqrt")
noise_cs=(1.0 1.2 1.4)

# Total combinations: 4 (ca.mode) * 4 (nco.mode) * 3 (ca_c) * 3 (nco_c) = 144
# Map SLURM_ARRAY_TASK_ID (0-143) to parameter indices

TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate indices
ca_mode_index=$(( TASK_ID / 36 ))                    # 0-3
nco_mode_index=$(( (TASK_ID / 9) % 4 ))              # 0-3
ca_noise_index=$(( (TASK_ID / 3) % 3 ))             # 0-2
nco_noise_index=$(( TASK_ID % 3 ))                  # 0-2

# Assign parameters based on indices
ca_mode=${ca_modes[$ca_mode_index]}
nco_mode=${nco_modes[$nco_mode_index]}
ca_c=${noise_cs[$ca_noise_index]}
nco_c=${noise_cs[$nco_noise_index]}

# Set a unique experiment name incorporating parameter values
EXP_NAME=eval_sweep_a0/ca${ca_mode}_c${ca_c}_nco${nco_mode}_c${nco_c}

# Run evaluation script with the selected parameters
python3 allatom_design/eval/sampling/draw_bb_samples.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=null \
    batch_size=64 \
    exp_name=${EXP_NAME} \
    ad_ckpt=${AD_CKPT} \
    length_range=[400,400] \
    length_step_size=1 \
    n_samples_per_length=100 \
    n_traj_per_length=0 \
    timestep_schedule.num_steps=500 \
    timestep_schedule.ca.mode=${ca_mode} \
    timestep_schedule.nco.mode=${nco_mode} \
    ca_diffusion.noise_schedule.name=step_scale \
    +ca_diffusion.noise_schedule.c=${ca_c} \
    nco_diffusion.noise_schedule.name=step_scale \
    +nco_diffusion.noise_schedule.c=${nco_c} \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
