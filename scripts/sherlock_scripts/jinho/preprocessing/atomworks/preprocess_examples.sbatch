#!/bin/bash
#SBATCH --job-name=preprocess_examples_atomworks_pdb_full
#SBATCH -p normal,owners
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --mem=32GB
#
# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A_%a.err
#
## Write batch script to logs
scontrol write batch_script $SLURM_JOB_ID /scratch/users/zhkim216/slurm_log_files/job_$SLURM_JOB_ID.sh

BASE_DIR=/home/users/zhkim216/code/allatom-design

cd ${BASE_DIR}

# Simplest: num_shards = MAX - MIN + 1
MIN=${SLURM_ARRAY_TASK_MIN:-0}
MAX=${SLURM_ARRAY_TASK_MAX:-0}
NUM_SHARDS=$(( MAX - MIN + 1 ))
echo "[sbatch] NUM_SHARDS=$NUM_SHARDS (from array [$MIN-$MAX])"

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"
BASE_DIR=/home/users/zhkim216/code/allatom-design

cd ${BASE_DIR}
# Run script
python3 allatom_design/data/preprocessing/atomworks/preprocess_examples.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=preprocess_examples \
    mmcif_dir=${PDB_MIRROR_PATH} \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v5 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=${NUM_SHARDS} \
    shard_id=${SLURM_ARRAY_TASK_ID} \
    batch_size=128 \
    max_tasks_per_child=16
