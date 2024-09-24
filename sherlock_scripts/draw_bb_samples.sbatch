#!/bin/bash
#SBATCH --job-name=a7_cbb14_bb_sampling
#SBATCH -p bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh

conda activate allatom_design
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

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/draw_bb_samples.py \
    num_workers=${SLURM_CPUS_ON_NODE} \
    mpnn.mpnn_params_dir=${MPNN_PARAMS_DIR} \
    nntm_dataset=null \
    batch_size=64 \
    exp_name=test \
    ad_ckpt=${AD_CKPT} \
    length_range=[400,400] \
    length_step_size=1 \
    n_samples_per_length=100 \
    n_traj_per_length=0 \
    timestep_schedule.num_steps=500 \
    timestep_schedule.ca.mode=linear \
    timestep_schedule.nco.mode=linear \
    ca_diffusion.noise_schedule.name=step_scale \
    +ca_diffusion.noise_schedule.c=1.0 \
    nco_diffusion.noise_schedule.name=step_scale \
    +nco_diffusion.noise_schedule.c=1.0 \
    +cond_labels.designability=DESIGNABLE \
    hydra.run.dir=${HYDRA_DIR}
