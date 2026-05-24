#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_setup_phelix.sh"

IMG="${SIF:-/scratch/users/zhkim216/containers/phelix.sif}"
REPO_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/phelix}"
ENV_DIR="${VENV:-/scratch/users/zhkim216/venv/phelix}"

if [ -x /bin/singularity ]; then
  APPTAINER_BIN="${APPTAINER_BIN:-/bin/singularity}"
else
  APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <sbatch_script.sbatch>" >&2
  exit 1
fi

JOB="$1"
[[ -f "$JOB" ]] || { echo "Not found: $JOB" >&2; exit 1; }

JOB_DIR="$(cd "$(dirname "$JOB")" && pwd)"
JOB_BASE="$(basename "$JOB")"
JOB_ABS="$JOB_DIR/$JOB_BASE"

WRAP_DIR="${SCRATCH:-/tmp}/slurm_phelix_container_wrappers"
mkdir -p "$WRAP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP="$WRAP_DIR/${JOB_BASE%.sbatch}.phelix.container.${STAMP}.sbatch"
ORIG_COPY="$WRAP_DIR/${JOB_BASE%.sbatch}.original.${STAMP}.sbatch"
cp "$JOB_ABS" "$ORIG_COPY"

binds=()
add_bind() {
  if [ -n "${1:-}" ]; then
    binds+=("$1")
  fi
}

add_bind "$SCRATCH"
add_bind "${GROUP_HOME:-}"
add_bind "$REPO_DIR"
add_bind "$ENV_DIR"
add_bind "$JOB_DIR"
add_bind "$WRAP_DIR"
add_bind "$TORCH_HOME"
add_bind "$HF_HOME"
add_bind "$PIP_CACHE_DIR"
add_bind "$XDG_CACHE_HOME"
add_bind "$PYTHONPYCACHEPREFIX"
add_bind "$TORCHINDUCTOR_CACHE_DIR"
add_bind "$TRITON_CACHE_DIR"
add_bind "$TORCH_EXTENSIONS_DIR"
add_bind "$UV_CACHE_DIR"
add_bind "$UV_PYTHON_INSTALL_DIR"
add_bind "$JAX_COMPILATION_CACHE_DIR"
add_bind "${OAK_LIBS:-}"
if [ -n "${MACHINE_ID_FILE:-}" ] && [ -f "$MACHINE_ID_FILE" ]; then
  add_bind "$MACHINE_ID_FILE:/etc/machine-id"
fi
if [ -d "$CUDA_HOST" ]; then
  add_bind "$CUDA_HOST:$CUDA_HOME:ro"
fi

BIND_LIST="$(IFS=,; echo "${binds[*]}")"
CONTAINER_PATH="$ENV_DIR/bin:/hmmer/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

{
  echo '#!/usr/bin/env bash'
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  cat <<EOF
set -euo pipefail
echo "[phelix-container] image: $IMG"
echo "[phelix-container] binds: $BIND_LIST"
echo "[phelix-container] original script saved to: $ORIG_COPY"

source "$SCRIPT_DIR/env_setup_phelix.sh"

$APPTAINER_BIN exec --nv \\
  --bind "$BIND_LIST" \\
  --env PATH="$CONTAINER_PATH" \\
  --env PYTHONPATH="$REPO_DIR:\${PYTHONPATH:-}" \\
  --env CUDA_HOME="$CUDA_HOME" \\
  --env LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}" \\
  --env TRITON_LIBCUDA_PATH="$TRITON_LIBCUDA_PATH" \\
  --env LIBRARY_PATH="$TRITON_LIBCUDA_PATH:\${LIBRARY_PATH:-}" \\
  --env TORCH_HOME="$TORCH_HOME" \\
  --env HF_HOME="$HF_HOME" \\
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \\
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \\
  --env PYTHONPYCACHEPREFIX="$PYTHONPYCACHEPREFIX" \\
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \\
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \\
  --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \\
  --env UV_CACHE_DIR="$UV_CACHE_DIR" \\
  --env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \\
  --env XLA_FLAGS="$XLA_FLAGS" \\
  --env XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \\
  --env XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION" \\
  --env JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR" \\
  --env VENV="$ENV_DIR" \\
  --env SCRATCH="$SCRATCH" \\
  --env PROJECT_ROOT="$REPO_DIR" \\
  --env SCHRODINGER_LD_LIBS="${SCHRODINGER_LD_LIBS:-}" \\
  --env SCHRODINGER="${SCHRODINGER:-}" \\
  --env SCHROD_LICENSE_FILE="${SCHROD_LICENSE_FILE:-}" \\
  "$IMG" \\
  bash -lc "set -euo pipefail; source '$ENV_DIR/bin/activate'; cd '$REPO_DIR'; exec bash '$JOB_ABS'"
EOF
} > "$WRAP"

chmod +x "$WRAP"
echo "[wrapper] Original script saved to: $ORIG_COPY"
echo "[wrapper] Submitting: $WRAP"
sbatch "$WRAP"
