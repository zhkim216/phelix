#!/bin/bash
#SBATCH --job-name=rcsb
#SBATCH -p cpu_batch
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
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
BOLTZ_SCRIPTS_DIR=/home/talal/rshuai/boltz/scripts/process

cd ${BOLTZ_SCRIPTS_DIR}

# Run script
redis-server --dir /scratch/hielab/rshuai/datasets/boltz --dbfilename ccd.rdb --port 7777 &

python3 rcsb.py \
    --datadir /large_storage/hielab/rshuai/datasets/mmCIF \
    --cluster /scratch/hielab/rshuai/datasets/boltz_v2/preprocessing/clustering/clustering.json \
    --outdir /scratch/hielab/rshuai/datasets/boltz_v2/rcsb_processed_targets \
    --max-file-size 7000000 \
    --redis-port 7777
