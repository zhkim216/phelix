#!/bin/bash
#SBATCH --job-name=cache_residue_data
#SBATCH -p possu,normal,owners
#SBATCH --array=0-8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/slurm_log_files/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

## Load modules and set environment variables:
ENV_DIR=/scratch/users/zhkim216/venv
venv af3ad

BASE_DIR=/home/users/zhkim216/code/allatom-design

cd ${BASE_DIR}

# Run script
python3 allatom_design/data/preprocessing/atomworks/cache_residue_data.py \
    --config-dir allatom_design/configs/data/preprocessing/atomworks \
    --config-name cache_residue_data.yaml \    
    seed=6 \
    out_dir=/scratch/users/zhkim216/datasets/atomworks \
    ccd_dir=/scratch/users/zhkim216/datasets/ccd_mirror \
    num_workers=${SLURM_CPUS_PER_TASK} \
    shard_id=${SLURM_ARRAY_TASK_ID} \
    num_shards=100 \
    n_trials=3 \
    timeout_offset=3.0 \
    timeout_slope=1.0 \
    num_conformers=50 \
    generate_conformers_kwargs.optimize=true \
    generate_conformers_kwargs.num_threads=2 \
    generate_conformers_kwargs.hydrogen_policy=remove
    