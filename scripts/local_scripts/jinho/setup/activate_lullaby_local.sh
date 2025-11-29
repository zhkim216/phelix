#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Activation Helper
# =============================================================================

ENV_NAME="lullaby_local"

# 1. Check if user activated the conda environment
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Warning: '$ENV_NAME' environment is NOT active."
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: mamba activate $ENV_NAME"
    # Don't exit, just warn
fi

echo "Configuring lullaby_local environment variables..."

# 2. Add HMMER to PATH
export PATH="$HOME/jinho/hmmer/bin:$PATH"

# 3. GTX1080 Compatibility Settings
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export JAX_LOG_COMPILES=1
export TORCH_COMPILE=0

echo "✓ Environment variables set for GTX1080"
echo "✓ HMMER path configured"
echo ""
echo "To test run:"
echo "  python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_seq_denoiser_local.yaml"
