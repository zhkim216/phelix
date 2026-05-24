#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_setup_elix.sh"

usage() {
  echo "Usage: $(basename "$0") [--schrodinger]" >&2
  echo "  --schrodinger  Bind Schrodinger support paths and license env." >&2
}

ENABLE_SCHRODINGER="${ELIX_ENABLE_SCHRODINGER:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schrodinger)
      ENABLE_SCHRODINGER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ "$ENABLE_SCHRODINGER" = "1" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/schrodinger_env.sh"
fi

if [ -x /bin/singularity ]; then
  APPTAINER_BIN="${APPTAINER_BIN:-/bin/singularity}"
else
  APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
fi

if [ ! -f "$SIF" ]; then
  echo "Container image not found: $SIF" >&2
  echo "Create it on Sherlock first, for example:" >&2
  echo "  mkdir -p $SCRATCH/containers" >&2
  echo "  apptainer pull $SIF docker://<your-elix-image>" >&2
  exit 2
fi

binds=()
add_bind() {
  if [ -n "${1:-}" ]; then
    binds+=("$1")
  fi
}

add_bind "$SCRATCH"
add_bind "$PROJECT_ROOT"
add_bind "$UV_CACHE_DIR"
add_bind "$UV_PYTHON_INSTALL_DIR"
add_bind "$TORCH_HOME"
add_bind "$HF_HOME"
add_bind "$PIP_CACHE_DIR"
add_bind "$XDG_CACHE_HOME"
add_bind "$PYTHONPYCACHEPREFIX"
add_bind "$TORCHINDUCTOR_CACHE_DIR"
add_bind "$TRITON_CACHE_DIR"
add_bind "$TORCH_EXTENSIONS_DIR"
add_bind "$JAX_COMPILATION_CACHE_DIR"
if [ -d "$CUDA_HOST" ]; then
  add_bind "$CUDA_HOST:$CUDA_HOME:ro"
fi
if [ "$ENABLE_SCHRODINGER" = "1" ]; then
  add_bind "${OAK_LIBS:-}"
  if [ -n "${MACHINE_ID_FILE:-}" ] && [ -f "$MACHINE_ID_FILE" ]; then
    add_bind "$MACHINE_ID_FILE:/etc/machine-id:ro"
  fi
fi

BIND_LIST="$(IFS=,; echo "${binds[*]}")"
CONTAINER_PATH="$VENV/bin:/hmmer/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
env_args=(
  --env "PATH=$CONTAINER_PATH"
  --env "PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}"
  --env "TORCH_HOME=$TORCH_HOME"
  --env "HF_HOME=$HF_HOME"
  --env "PIP_CACHE_DIR=$PIP_CACHE_DIR"
  --env "XDG_CACHE_HOME=$XDG_CACHE_HOME"
  --env "PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX"
  --env "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  --env "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  --env "UV_ENV_ROOT=$UV_ENV_ROOT"
  --env "UV_CACHE_DIR=$UV_CACHE_DIR"
  --env "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  --env "XLA_FLAGS=$XLA_FLAGS"
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  --env "XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION"
  --env "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
  --env "VENV=$VENV"
  --env "SCRATCH=$SCRATCH"
  --env "PROJECT_ROOT=$PROJECT_ROOT"
  --env "CUDA_HOME=$CUDA_HOME"
  --env "TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH"
  --env "LIBRARY_PATH=$TRITON_LIBCUDA_PATH:${LIBRARY_PATH:-}"
  --env "LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
)
if [ "$ENABLE_SCHRODINGER" = "1" ]; then
  env_args+=(
    --env "SCHRODINGER_LD_LIBS=${SCHRODINGER_LD_LIBS:-}"
    --env "SCHRODINGER=${SCHRODINGER:-}"
    --env "SCHROD_LICENSE_FILE=${SCHROD_LICENSE_FILE:-}"
  )
fi

echo "Starting Elix interactive shell in container"
echo "============================================="
echo "Image: $SIF"
echo "Python: $VENV/bin/python"
echo "Project: $PROJECT_ROOT"
echo "CUDA: $CUDA_HOME"
echo "Schrodinger: $ENABLE_SCHRODINGER"
echo "============================================="

unset PROMPT_COMMAND

"$APPTAINER_BIN" shell --nv \
  --bind "$BIND_LIST" \
  "${env_args[@]}" \
  "$SIF"
