#!/bin/bash
#SBATCH --job-name=preprocess_examples
#SBATCH -p possu,normal,owners
#SBATCH --array=0-19
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
ENV_DIR=/scratch/users/rshuai/envs
source ${ENV_DIR}/allatom_design/bin/activate

export PDB_MIRROR_PATH=""
export CCD_MIRROR_PATH="/scratch/users/rshuai/datasets/ccd"
BASE_DIR=/home/users/rshuai/code/allatom-design

cd ${BASE_DIR}
# Run script
python3 allatom_design/data/preprocessing/atomworks/preprocess_examples.py \
    mmcif_dir=/scratch/users/rshuai/datasets/mmCIF \
    out_dir=/scratch/users/rshuai/datasets/atomworks \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=20 \
    shard_id=${SLURM_ARRAY_TASK_ID}
