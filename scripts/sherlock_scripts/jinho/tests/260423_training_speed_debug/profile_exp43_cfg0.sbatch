#!/usr/bin/env bash
# profile_exp43_cfg0.sbatch
# exp43_cfg0 full config (batch=8, max_tokens=512, max_atoms=4608) 에서 training step의
# phase-별 wall time (dataload / forward / backward / optim) 을 측정한다.
# torch.compile on, bf16-mixed autocast, Chrome trace (profile_trace.json) 출력.
# 목적: Tier-0/Tier-1 패치 적용 후 실제 bottleneck 이 무엇인지 확인.
#SBATCH --job-name=profile_exp43_cfg0
#SBATCH -p possu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/profile_exp43_cfg0_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/profile_exp43_cfg0_%j.err

BASE_DIR=/home/users/zhkim216/code/allatom-design
OUT_BASE=/scratch/users/zhkim216/profile_outputs/260423_training_speed_debug
mkdir -p "${OUT_BASE}" /scratch/users/zhkim216/job_output/train_seq_denoiser

export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_profile_cfg0"
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
    --pdb-path "${PDB_PATH}" \
    --metadata-path "${METADATA_PATH}" \
    --residue-cache-dir "${RESIDUE_CACHE_DIR}" \
    --validation-ids-file "${VAL_IDS}" \
    --val-metadata-path "${VAL_METADATA}" \
    --n-warmup 20 \
    --n-steps 100 \
    --accum-grad-batches 4 \
    --output-csv "${OUT_BASE}/profile_cfg0_full.csv" \
    --trace-path "${OUT_BASE}/profile_cfg0_full_trace.json" \
    --torch-logs
