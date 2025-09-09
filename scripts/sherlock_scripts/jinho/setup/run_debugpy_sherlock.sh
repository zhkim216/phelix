#!/bin/bash
# Scripts for vscode debugger

# env_setup.sh
source "/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup/env_setup.sh"

echo "[Running debugpy in container]"
echo "Script: $1"
echo "CUDA_HOME: $CUDA_HOME"
echo "Working directory: $(pwd)"

/bin/singularity exec --nv \
  --bind "$SCRATCH","$UV_CACHE_DIR:/uv/cache","$UV_PYTHON_INSTALL_DIR:/uv/python" \
  --bind "$CUDA_HOME:$CUDA_HOME" \
  --env CUDA_HOME=$CUDA_HOME \
  --env PATH=$CUDA_HOME/bin:$PATH \
  --env LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
  --env XLA_FLAGS="$XLA_FLAGS" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
  --env XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION \
  "$SIF" \
  $VENV/bin/python -m debugpy --listen 127.0.0.1:5678 --wait-for-client "$1"