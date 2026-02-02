#!/bin/bash
# Scripts for vscode debugger (lullaby env)

# 1. 환경 설정 로드 (여기서 SIF, VENV, OST 경로 등이 잡힘)
source "/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup/env_setup.sh"

echo "[Running debugpy in container: lullaby]"
echo "Target Script: $1"
echo "SIF Image: $SIF"
echo "VENV Path: $VENV"

# 2. Apptainer/Singularity 실행
# - OST 경로(dist-packages)를 PYTHONPATH에 추가
# - VENV 내 python 사용
/bin/singularity exec --nv \
  --bind "$SCRATCH","$UV_CACHE_DIR:/uv/cache","$UV_PYTHON_INSTALL_DIR:/uv/python" \
  --env CUDA_HOME=$CUDA_HOME \
  --env PATH=$VENV/bin:$CUDA_HOME/bin:$PATH \
  --env LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
  --env PYTHONPATH=$PROJECT_ROOT:/usr/lib/python3/dist-packages:$PYTHONPATH \
  --env OST_COMPOUND_LIB=$OST_COMPOUND_LIB \
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
  "$SIF" \
  $VENV/bin/python -m debugpy --listen 127.0.0.1:5678 --wait-for-client "$1"