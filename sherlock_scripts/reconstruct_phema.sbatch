#!/bin/bash
#SBATCH --job-name=phema
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/rshuai/subhang/slurm_log_files/job_%j.out
#SBATCH --error=/scratch/users/rshuai/subhang/slurm_log_files/job_%j.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/rshuai/subhang/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and conda environment
source ~/.subhang_bash_profile
ml python
module load gcc/12.4.0
module load cuda/12.1
source /home/groups/possu/miniconda/etc/profile.d/conda.sh
conda activate allatom_design

# Define directories
BASE_DIR=/home/users/rshuai/subhang/allatom-design
HYDRA_DIR=/scratch/users/rshuai/subhang/hydra_outputs

cd ${BASE_DIR}

# Model name to evaluate
MODEL_NAME=baseline_fs512_bs128
MODEL_TYPE=atom_denoiser  # atom_denoiser or seq_denoiser

# Directory where denoiser was trained to
DENOISER_TRAIN_DIR=/scratch/users/rshuai/subhang/allatom_design/allatom_design/${MODEL_NAME}

# Run eval script
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/reconstruct_phema.py \
    denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    model_type=${MODEL_TYPE} \
    out_step=500000 \
    out_stds=null \
    hydra.run.dir=${HYDRA_DIR}
