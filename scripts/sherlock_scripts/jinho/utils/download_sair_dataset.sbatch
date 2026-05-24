#!/bin/bash
#SBATCH --job-name=sair_download
#SBATCH -p possu,normal,owners
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/utils/sair_download_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/utils/sair_download_%j.err

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
DOWNLOAD_DIR=/scratch/users/zhkim216/sair

mkdir -p /scratch/users/zhkim216/job_output/utils "${DOWNLOAD_DIR}"

cd "${BASE_DIR}"

echo "============================================"
echo "SAIR Dataset Download"
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Download dir: ${DOWNLOAD_DIR}"
echo "============================================"

python3 scripts/sherlock_scripts/jinho/utils/download_sair_dataset.py \
    --download-dir "${DOWNLOAD_DIR}"

echo "============================================"
echo "SAIR download complete"
echo "End time: $(date)"
echo "============================================"
