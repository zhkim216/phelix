#!/usr/bin/env bash
set -euo pipefail

SIF="/scratch/users/zhkim216/containers/allscaip.sif"
VENV="/scratch/users/zhkim216/venv/allscaip"
SING_HOME="$SCRATCH/.singhome"
mkdir -p "$SING_HOME"

# caches
export TORCH_HOME="$SCRATCH/cache/torch"
export HF_HOME="$SCRATCH/cache/huggingface"
export XDG_CACHE_HOME="$SCRATCH/cache/.cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/cache/inductor_cache"
export TRITON_CACHE_DIR="$SCRATCH/cache/triton_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH/cache/torch_extensions"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

[[ -f "$SIF" ]] || { echo "SIF not found: $SIF"; exit 1; }

# venv 점검/자동 생성 (컨테이너 내부에서)
apptainer exec --nv --cleanenv --home "$SING_HOME" \
  --bind "$SCRATCH:$SCRATCH" "$SIF" /usr/bin/bash -lc "
  set -e
  if ! \"$VENV/bin/python\" -V >/dev/null 2>&1; then
    echo '[INFO] creating venv at $VENV'
    /usr/bin/python3 -m venv --system-site-packages \"$VENV\"
    \"$VENV/bin/pip\" install -U pip wheel setuptools
  fi
"

# 인터랙티브 진입 (/usr/bin/bash 고정)
exec apptainer exec --nv --cleanenv --home "$SING_HOME" \
  --bind "$SCRATCH:$SCRATCH" "$SIF" /usr/bin/bash -lc "
  export PIP_INDEX_URL='https://pypi.org/simple'
  export PIP_EXTRA_INDEX_URL='https://download.pytorch.org/whl/cu126'
  export TORCH_HOME='$TORCH_HOME'
  export HF_HOME='$HF_HOME'
  export XDG_CACHE_HOME='$XDG_CACHE_HOME'
  export TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR'
  export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
  export TORCH_EXTENSIONS_DIR='$TORCH_EXTENSIONS_DIR'
  export PATH='$VENV/bin':\$PATH
  echo 'Inside container. Python:' \$(python -V)
  exec bash --noprofile --norc
"
