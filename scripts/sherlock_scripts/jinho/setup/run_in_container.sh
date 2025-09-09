#!/bin/bash
# Helper script for singularity container execution

# Find env_setup.sh
source "/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup/env_setup.sh"

# Execution
/bin/singularity exec --nv \
    --bind "$SCRATCH","$UV_CACHE_DIR:/uv/cache","$UV_PYTHON_INSTALL_DIR:/uv/python" \
    --bind "$CUDA_HOME:$CUDA_HOME" \
    --bind "$CUDA_HOME:/usr/local/cuda" \
    --env CUDA_HOME=$CUDA_HOME \
    --env PATH=$VENV/bin:$CUDA_HOME/bin:$PATH \
    --env LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    --env PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH \
    --env TORCH_HOME=$TORCH_HOME \
    --env HF_HOME=$HF_HOME \
    --env PIP_CACHE_DIR=$PIP_CACHE_DIR \
    --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
    --env PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX \
    --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    --env TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR \
    --env UV_CACHE_DIR=$UV_CACHE_DIR \
    --env UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR \
    --env XLA_FLAGS="$XLA_FLAGS" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
    --env XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION \
    --env JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR \
    --env VENV=$VENV \
    --env SCRATCH=$SCRATCH \    
    "$SIF" $VENV/bin/python "$@"