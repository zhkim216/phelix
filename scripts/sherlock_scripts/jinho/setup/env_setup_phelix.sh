#!/usr/bin/env bash

# Sherlock environment wiring for the AlphaFold3-new Phelix setup.
# Source this on Sherlock before entering the container or wrapping sbatch jobs.

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.1 || true
fi

# Cache and build locations.
export SCRATCH="${SCRATCH:-/scratch/users/zhkim216}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/cache/torch}"
export HF_HOME="${HF_HOME:-$SCRATCH/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/cache/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/cache/.cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$SCRATCH/cache/.pycache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH/cache/inductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/cache/triton_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SCRATCH/cache/torch_extensions}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRATCH/uv/python}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/cache/jax_compilation_cache}"

mkdir -p \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$PYTHONPYCACHEPREFIX" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$JAX_COMPILATION_CACHE_DIR"

# CUDA setup. CUDA_HOST is the host module path; CUDA_HOME is the in-container path.
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOST="${CUDA_HOST:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi
export CUDA_HOST="${CUDA_HOST:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [ -f "$CUDA_HOME/targets/x86_64-linux/lib/stubs/libcuda.so" ]; then
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME/targets/x86_64-linux/lib/stubs}"
else
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME/lib64/stubs}"
fi

# AlphaFold3/JAX settings. The disable_hlo_passes flag is needed on V100/T4-class GPUs.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_disable_hlo_passes=custom-kernel-fusion-rewriter}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"

# Phelix paths.
export SIF="${SIF:-$SCRATCH/containers/phelix.sif}"
export VENV="${VENV:-$SCRATCH/venv/phelix}"
export PROJECT_ROOT="${PROJECT_ROOT:-/home/users/zhkim216/code/allatom-design}"

# Optional Schrodinger environment. Keep optional so Phelix AF3 setup does not
# depend on Schrodinger for package installation.
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SETUP_DIR/schrodinger_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SETUP_DIR/schrodinger_env.sh" || true
fi

export PATH="$VENV/bin:/hmmer/bin:$CUDA_HOME/bin:$PATH"

echo "Phelix Sherlock environment loaded:"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_HOST: $CUDA_HOST"
echo "  SIF: $SIF"
echo "  VENV: $VENV"
