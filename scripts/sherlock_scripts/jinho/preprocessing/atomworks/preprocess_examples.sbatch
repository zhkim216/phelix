#!/bin/bash
#SBATCH --job-name=preprocess_examples_atomworks_pdb_full_v8
#SBATCH -p possu,normal,owners
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0:00:00
#SBATCH --mem=32GB

# Send stdout and stderr to logs
#SBATCH --output=/scratch/users/zhkim216/job_output/data_preprocessing/preprocess_examples_v8_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/data_preprocessing/preprocess_examples_v8_%A_%a.err

BASE_DIR=/home/users/zhkim216/code/allatom-design

export PDB_MIRROR_PATH="/scratch/users/zhkim216/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/scratch/users/zhkim216/datasets/ccd_mirror"

mkdir -p /scratch/users/zhkim216/job_output/data_preprocessing
cd ${BASE_DIR}

# Simplest: num_shards = MAX - MIN + 1
MIN=${SLURM_ARRAY_TASK_MIN:-0}
MAX=${SLURM_ARRAY_TASK_MAX:-0}
NUM_SHARDS=$(( MAX - MIN + 1 ))
echo "Starting shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} for atomworks_pdb_full_v8 (array [$MIN-$MAX])"

# Run script
python3 allatom_design/data/preprocessing/atomworks/preprocess_examples.py \
    --config-path=/home/users/zhkim216/code/allatom-design/allatom_design/configs/data/preprocessing/atomworks \
    --config-name=preprocess_examples \
    mmcif_dir=${PDB_MIRROR_PATH} \
    out_dir=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8 \
    num_workers=${SLURM_CPUS_PER_TASK} \
    num_shards=${NUM_SHARDS} \
    shard_id=${SLURM_ARRAY_TASK_ID}

echo "Completed shard ${SLURM_ARRAY_TASK_ID}"
