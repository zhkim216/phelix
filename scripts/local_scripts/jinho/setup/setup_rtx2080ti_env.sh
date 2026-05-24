#!/bin/bash
# Environment setup for AF3 + allatom-design on the RTX 2080 Ti desktop

echo "=== RTX 2080 Ti (Turing Architecture) Environment Setup ==="

# Activate virtual environment
# Note: User should activate mamba environment manually
if [ "$CONDA_DEFAULT_ENV" != "elix_local" ]; then
    echo "Warning: 'elix_local' environment is not active."
fi

# XLA settings for compute capability 7.x compatibility (REQUIRED for AF3)
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"

# GPU memory settings (unified memory mode)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8  # RTX 2080 Ti has 11 GB VRAM

# CUDA device configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# JAX logging (for debugging)
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-0}"

# PyTorch settings (disable torch.compile)
export TORCH_COMPILE=0
export AF3_FLASH_ATTENTION_IMPLEMENTATION=xla

echo "Environment variables configured successfully!"
echo "XLA_FLAGS: $XLA_FLAGS"
echo "GPU memory: Unified memory mode"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Important notes:"
echo "- For AF3 on compute capability 7.x, use --flash_attention_implementation=xla"
echo "- Use small batch sizes (limited GPU memory)"
echo "- Test complex models on CPU environment first"
echo ""
echo "Test command:"
echo "python -c \"import torch, jax; print('PyTorch CUDA:', torch.cuda.is_available()); print('JAX devices:', jax.devices())\""
