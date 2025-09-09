#!/bin/bash
# scripts for setup environment for AF3ppg container

# Load cuda module
module load cuda/12.6.1

# Cache & builds
export TORCH_HOME=/scratch/users/zhkim216/cache/torch
export HF_HOME=/scratch/users/zhkim216/cache/huggingface
export PIP_CACHE_DIR=/scratch/users/zhkim216/cache/pip_cache
export XDG_CACHE_HOME=/scratch/users/zhkim216/cache/.cache
export PYTHONPYCACHEPREFIX=/scratch/users/zhkim216/cache/.pycache
export TORCHINDUCTOR_CACHE_DIR=/scratch/users/zhkim216/cache/inductor_cache
export TRITON_CACHE_DIR=/scratch/users/zhkim216/cache/triton_cache
export TORCH_EXTENSIONS_DIR=/scratch/users/zhkim216/cache/torch_extensions
export UV_CACHE_DIR=/scratch/users/zhkim216/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/users/zhkim216/uv/python
export JAX_COMPILATION_CACHE_DIR=/scratch/users/zhkim216/cache/jax_compilation_cache

# CUDA setups
export CUDA_HOME=/share/software/user/open/cuda/12.6.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# XLA/JAX setups (AF3 recommendation)
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

# Paths
export SIF=/scratch/users/zhkim216/containers/af3ad_base.sif
export VENV=/scratch/users/zhkim216/venv/af3ad
export SCRATCH=/scratch/users/zhkim216

# Project root (Parent directory of the scripts)
export PROJECT_ROOT="$(dirname $(dirname $(readlink -f ${BASH_SOURCE[0]})))"

echo "Environment loaded:"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  SIF: $SIF"
echo "  VENV: $VENV"