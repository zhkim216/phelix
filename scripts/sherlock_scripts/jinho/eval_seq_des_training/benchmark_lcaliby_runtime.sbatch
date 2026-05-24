#!/bin/bash
#SBATCH --job-name=bench_lcaliby
#SBATCH -p possu,bioe
#SBATCH --array=0-1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/bench_lcaliby_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/bench_lcaliby_%A_%a.err

# Directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
CIF_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs
OUTPUT_DIR=/scratch/users/zhkim216/out_dir/runtime_benchmark
RESIDUE_CACHE=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data

mkdir -p /scratch/users/zhkim216/job_output "${OUTPUT_DIR}"

cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

# Array: 0=exp35_cfg2, 1=exp37_cfg11
EXP_NAMES=("exp35_cfg2" "exp37_cfg11")
CKPT_PATHS=(
    "/scratch/users/zhkim216/out_dir/ligand_seq_des/exp35_cfg2/checkpoints/ema/sd-step82500-epoch52-ema0.99.ckpt"
    "/scratch/users/zhkim216/out_dir/ligand_seq_des/exp37_cfg11/checkpoints/ema/sd-step22500-epoch14-ema0.99.ckpt"
)

EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT_PATH=${CKPT_PATHS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "EXP_NAME: ${EXP_NAME}"
echo "CKPT_PATH: ${CKPT_PATH}"
echo "=========================================="

python3 allatom_design/eval/benchmarking/benchmark_lcaliby_runtime.py \
    --exp_name "${EXP_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --cif_dir "${CIF_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --residue_cache_dir "${RESIDUE_CACHE}"

echo "Benchmark complete for ${EXP_NAME}"
