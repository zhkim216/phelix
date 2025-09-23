#!/bin/bash
# Generic Slurm payload wrapper to run commands inside a Singularity container
# - Use with: sbatch [slurm options] path/to/sbatch_container.sh -- <command-and-args>
# - By default, runs the command via the Python from $VENV inside the container.
#   Set CONTAINER_MODE=exec to run a raw command instead of python.

set -euo pipefail

# Resolve this script directory and env setup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SETUP="${ENV_SETUP:-${SCRIPT_DIR}/env_setup.sh}"

if [[ ! -f "$ENV_SETUP" ]]; then
  echo "[error] env_setup.sh not found at $ENV_SETUP" 1>&2
  exit 1
fi

# Load environment (modules, paths, SIF, VENV, SCRATCH, CUDA, caches ...)
source "$ENV_SETUP"

# Parse mode: python (default) or exec
MODE="${CONTAINER_MODE:-python}"

# Expect a separator then the command we will run inside the container
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch [slurm opts] $0 [--mode python|exec] -- <command and args>" 1>&2
  echo "Example (python): sbatch -J job -p owners -t 12:00:00 -c 8 $0 -- allatom_design/script.py --foo bar" 1>&2
  echo "Example (exec):   CONTAINER_MODE=exec sbatch -J job -p owners -t 01:00:00 $0 -- nvidia-smi" 1>&2
  exit 2
fi

CMD=("$@")

echo "[container wrapper] Node: $(hostname)"
echo "[container wrapper] SLURM job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID cpus=$SLURM_CPUS_PER_TASK"
echo "[container wrapper] Mode: $MODE"
echo "[container wrapper] Command: ${CMD[*]}"

if [[ ! -f "$SIF" ]]; then
  echo "[error] SIF image not found: $SIF" 1>&2
  exit 3
fi

# Build entrypoint
ENTRY=( )
if [[ "$MODE" == "python" ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[error] Python not found at $VENV/bin/python (set VENV in env_setup.sh)" 1>&2
    exit 4
  fi
  ENTRY=("$VENV/bin/python")
fi

# Execute inside the container
exec /bin/singularity exec --nv \
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
  "$SIF" "${ENTRY[@]}" "${CMD[@]}"


