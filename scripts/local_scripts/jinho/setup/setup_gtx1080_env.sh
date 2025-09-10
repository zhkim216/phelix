#!/bin/bash
# Environment setup for AF3 + allatom-design on GTX1080

echo "=== GTX1080 (Pascal Architecture) Environment Setup ==="

# Activate virtual environment
source ~/venv/af3ad_desktop/bin/activate

# XLA settings for GTX1080 compatibility (REQUIRED!)
# CC 6.1 (Pascal) has similar issues to CC 7.x, so we apply the same workaround
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"

# GPU memory settings (unified memory mode)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8  # Considering GTX1080 8GB VRAM

# CUDA device configuration
export CUDA_VISIBLE_DEVICES=0

# JAX logging (for debugging)
export JAX_LOG_COMPILES=1

# PyTorch settings (disable torch.compile)
export TORCH_COMPILE=0

echo "Environment variables configured successfully!"
echo "XLA_FLAGS: $XLA_FLAGS"
echo "GPU memory: Unified memory mode"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Important notes:"
echo "- Do NOT use torch.compile() (unsupported on Pascal architecture)"
echo "- Use small batch sizes (limited GPU memory)"
echo "- Test complex models on CPU environment first"
echo ""
echo "Test command:"
echo "python -c \"import torch, jax; print('PyTorch CUDA:', torch.cuda.is_available()); print('JAX devices:', jax.devices())\""
