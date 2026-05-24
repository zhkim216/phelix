#!/bin/bash
#SBATCH --job-name=bench_lmpnn
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/bench_lmpnn_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/bench_lmpnn_%j.err

# Directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
LMPNN_DIR=/home/users/zhkim216/code/LigandMPNN
CIF_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
OUTPUT_DIR=/scratch/users/zhkim216/out_dir/runtime_benchmark

mkdir -p /scratch/users/zhkim216/job_output "${OUTPUT_DIR}"

cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

echo "=========================================="
echo "LMPNN_DIR: ${LMPNN_DIR}"
echo "CIF_DIR: ${CIF_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "=========================================="

python3 allatom_design/eval/benchmarking/benchmark_ligandmpnn_runtime.py \
    --lmpnn_dir "${LMPNN_DIR}" \
    --cif_dir "${CIF_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Benchmark complete for LigandMPNN"
