#!/usr/bin/env bash
# profile_speedup.sbatch
# exp43_speedup.sbatch 와 동일한 config (batch=16, accumulate=2) 로 profile 돌려서
# Tier-0/1 패치의 실제 step time 개선폭을 정량화한다.
# 동일 effective batch 유지 (16*2 = 32 vs cfg0의 8*4=32)
#SBATCH --job-name=profile_speedup
#SBATCH -p possu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/profile_speedup_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/profile_speedup_%j.err

BASE_DIR=/home/users/zhkim216/code/allatom-design
OUT_BASE=/scratch/users/zhkim216/profile_outputs/260423_training_speed_debug
mkdir -p "${OUT_BASE}" /scratch/users/zhkim216/job_output/train_seq_denoiser

export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_profile_speedup"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

PDB_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3
METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v3/metadata_seq_clustered_04_filtered_grouped_250205.parquet
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
VAL_IDS=/scratch/users/zhkim216/datasets/splits/lmpnn_validation_ids.txt
VAL_METADATA=/scratch/users/zhkim216/datasets/val_cifs/lmpnn_val_cifs/metadata_lmpnnval_small_molecule_filtered_for_training.parquet

HYDRA_FULL_ERROR=1 \
python3 allatom_design/tests/260423_training_speed_debug/profile_training.py \
    --config full --compile \
    --batch-size 16 \
    --pdb-path "${PDB_PATH}" \
    --metadata-path "${METADATA_PATH}" \
    --residue-cache-dir "${RESIDUE_CACHE_DIR}" \
    --validation-ids-file "${VAL_IDS}" \
    --val-metadata-path "${VAL_METADATA}" \
    --n-warmup 20 \
    --n-steps 100 \
    --accum-grad-batches 2 \
    --output-csv "${OUT_BASE}/profile_speedup.csv" \
    --trace-path "${OUT_BASE}/profile_speedup_trace.json" \
    --torch-logs
