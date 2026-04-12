#!/bin/bash
# Interactive shell in Singularity container

# Find env_setup.sh and source it
source "/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup/env_setup.sh"

echo "Starting interactive shell in Singularity container..."
echo "=========================================="
echo "Python: $VENV/bin/python"
echo "Project: $PROJECT_ROOT"
echo "CUDA: $CUDA_HOME"
echo "To exit: type 'exit' or Ctrl+D"
echo "=========================================="

# Clear VSCode-specific environment variables
unset PROMPT_COMMAND

# For torch.compile
CUDA_BIND_OPT="--bind ${CUDA_HOST}:${CUDA_HOME}:ro"

# Start interactive shell
/bin/singularity shell --nv \
    $CUDA_BIND_OPT \
    --bind "$SCRATCH" \
    --bind "$PROJECT_ROOT" \
    --bind "$UV_CACHE_DIR:/uv/cache" \
    --bind "$UV_PYTHON_INSTALL_DIR:/uv/python" \
    --env PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH \
    --env PS1="\[\033[01;35m\][singularity]\[\033[00m\] \[\033[01;34m\]\w\[\033[00m\] $ " \
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
    --env PROJECT_ROOT=$PROJECT_ROOT \
    --env CUDA_HOME=$CUDA_HOME \
    --env TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH \
    --env LIBRARY_PATH=$TRITON_LIBCUDA_PATH:$LIBRARY_PATH \
    --env LD_LIBRARY_PATH=$SCHRODINGER_LD_LIBS:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64 \
    --env SCHRODINGER=$SCHRODINGER \
    --env SCHROD_LICENSE_FILE=$SCHROD_LICENSE_FILE \
    --bind "$OAK_LIBS" \
    --bind "$MACHINE_ID_FILE:/etc/machine-id:ro" \
    "$SIF"