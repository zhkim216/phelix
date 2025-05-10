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
conda activate rshuai_allatom_design
BASE_DIR=/home/talal/rshuai/allatom-design

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/boltz_v2/process_pdbs.py \
    mmcif_dir=/large_storage/hielab/rshuai/datasets/mmCIF \
    out_dir=/scratch/hielab/rshuai/datasets/boltz_v2 \
    software_path=/scratch/hielab/rshuai/software \
    cluster_rdb_path=/scratch/hielab/rshuai/datasets/boltz_v2/preprocessing/clustering/clustering.rdb \
    num_workers=32
