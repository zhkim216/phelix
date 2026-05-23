#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Activation Helper
# =============================================================================

MODEL_DEV_ROOT="${MODEL_DEV_ROOT:-$HOME/model-dev}"
ENV_NAME="${ENV_NAME:-phelix_local}"
HMMER_INSTALL_DIR="${HMMER_INSTALL_DIR:-$MODEL_DEV_ROOT/software/hmmer-3.4-af3}"
DATASETS_ROOT="${DATASETS_ROOT:-$MODEL_DEV_ROOT/datasets}"

# 1. Check if user activated the conda environment
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Warning: '$ENV_NAME' environment is NOT active."
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: micromamba activate $ENV_NAME"
    # Don't exit, just warn
fi

echo "Configuring $ENV_NAME environment variables..."

# 2. Add HMMER to PATH
export PATH="$HMMER_INSTALL_DIR/bin:$PATH"

# 3. Local AtomWorks data mirrors.
export PDB_MIRROR_PATH="${PDB_MIRROR_PATH:-$DATASETS_ROOT/pdb_mirror}"
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-$DATASETS_ROOT/ccd_mirror}"

# 4. RTX 2080 Ti / Turing compatibility settings for AF3 and local training.
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"
export TORCH_COMPILE=0
export AF3_FLASH_ATTENTION_IMPLEMENTATION=xla

echo "✓ Environment variables set for RTX 2080 Ti"
echo "✓ HMMER path configured: $HMMER_INSTALL_DIR/bin"
echo "✓ PDB mirror path: $PDB_MIRROR_PATH"
echo "✓ CCD mirror path: $CCD_MIRROR_PATH"
echo ""
echo "To test run:"
echo "  python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_seq_denoiser_local.yaml"
echo ""
echo "For local AlphaFold3 inference on this Turing GPU, pass:"
echo "  --flash_attention_implementation=xla"
