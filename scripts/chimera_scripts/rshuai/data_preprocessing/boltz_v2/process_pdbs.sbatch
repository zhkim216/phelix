#!/bin/bash
#SBATCH --job-name=process_pdbs
#SBATCH -p cpu_batch
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
ENV_DIR=/scratch/hielab/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

BASE_DIR=/home/talal/rshuai/allatom-design
SOFTWARE_PATH=/scratch/hielab/rshuai/software
export SOFTWARE_PATH=${SOFTWARE_PATH}  # required for mmseqs, foldseek, redis, etc.

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/boltz_v2/process_pdbs.py \
    mmcif_dir=/large_storage/hielab/rshuai/datasets/mmCIF \
    out_dir=/scratch/hielab/rshuai/datasets/boltz_v2 \
    num_workers=32
