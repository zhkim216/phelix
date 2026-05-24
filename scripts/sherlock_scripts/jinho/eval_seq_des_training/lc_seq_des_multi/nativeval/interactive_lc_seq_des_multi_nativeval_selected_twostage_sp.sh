#!/bin/bash
# Interactive script for two-stage sequence+pocket design evaluation
# Model: exp35_cfg2 (seq denoiser) on exp37_cfg11 (struct gen) outputs
# Valset: nativeval_sm_selected_twostage_sp
#
# Usage on Sherlock interactive node:
#   srun -p possu --gpus-per-node=1 --cpus-per-task=8 --mem=16G --time=12:00:00 --pty bash
#   bash scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/nativeval/interactive_lc_seq_des_multi_nativeval_selected_twostage_sp.sh

set -euo pipefail

# Define directories
BASE_DIR=/home/users/zhkim216/code/allatom-design
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des

mkdir -p "${HYDRA_DIR}" "${BASE_OUT_DIR}"

cd "${BASE_DIR}"

# Ignore warnings
export PYTHONWARNINGS="ignore"

# Automatically detect flash attention implementation
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

echo "Starting interactive evaluation: exp35_cfg2 / nativeval_sm_selected_twostage_sp / step=82500"

# Run evaluation script
# Config: lc_seq_des_multi_debug (debug=true, num_debug_samples=2)
# To run full evaluation, add: debug=false num_workers=8
WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path ${BASE_DIR}/allatom_design/configs/eval/sampling \
    --config-name lc_seq_des_multi_debug \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete."
