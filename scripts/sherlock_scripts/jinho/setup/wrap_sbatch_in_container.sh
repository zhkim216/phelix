#!/usr/bin/env bash
set -euo pipefail

# Load environment setup first
SCRIPT_DIR="/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup"
source "${SCRIPT_DIR}/env_setup.sh"

# --- EDIT THESE THREE (cluster-specific) ---
IMG="${SIF:-/scratch/users/zhkim216/containers/lullaby.sif}"       # 컨테이너 이미지(.sif)
REPO_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/allatom-design}"  # 저장소 루트
ENV_DIR="${VENV:-/opt/lullaby}"                 # venv 디렉토리
# ------------------------------------------

# Pick runner (apptainer or singularity)
APPTAINER_BIN="${APPTAINER_BIN:-/bin/singularity}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <sbatch_script.sbatch>" >&2
  exit 1
fi

JOB="$1"
[[ -f "$JOB" ]] || { echo "Not found: $JOB" >&2; exit 1; }

# Resolve abs path of original sbatch
JOB_DIR="$(cd "$(dirname "$JOB")" && pwd)"
JOB_BASE="$(basename "$JOB")"
JOB_ABS="$JOB_DIR/$JOB_BASE"

# Where to place generated wrapper sbatch
WRAP_DIR="${SCRATCH:-/tmp}/slurm_container_wrappers"
mkdir -p "$WRAP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP="$WRAP_DIR/${JOB_BASE%.sbatch}.container.${STAMP}.sbatch"

# Binds
BIND=""
[[ -n "${SCRATCH:-}"    ]] && BIND="$BIND,$SCRATCH"
[[ -n "${GROUP_HOME:-}" ]] && BIND="$BIND,$GROUP_HOME"
BIND="$BIND,$REPO_DIR,$ENV_DIR,$JOB_DIR"
# Add cache directories from env_setup.sh
BIND="$BIND,$TORCH_HOME,$HF_HOME,$PIP_CACHE_DIR,$XDG_CACHE_HOME"
BIND="$BIND,$PYTHONPYCACHEPREFIX,$TORCHINDUCTOR_CACHE_DIR,$TRITON_CACHE_DIR"
BIND="$BIND,$TORCH_EXTENSIONS_DIR,$UV_CACHE_DIR,$UV_PYTHON_INSTALL_DIR"
BIND="$BIND,$JAX_COMPILATION_CACHE_DIR"

# CUDA bind, important for torch.compile
BIND="$BIND,${CUDA_HOST}:${CUDA_HOME}:ro"

# Capture full PATH for container (must be done after env_setup.sh)
# Include /hmmer/bin which is inside the container image
CONTAINER_PATH="$VENV/bin:$CUDA_HOME/bin:/hmmer/bin:$PATH"

{
  echo '#!/usr/bin/env bash'
  # Keep original #SBATCH
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  cat <<EOF
set -euo pipefail
echo "[container] image: $IMG"
echo "[container] binds: $BIND"

# Load env_setup.sh in the wrapper to get all environment variables
source "$SCRIPT_DIR/env_setup.sh"

$APPTAINER_BIN exec --nv \\
  --bind "$BIND" \\
  --env CUDA_HOME=$CUDA_HOME \\
  --env PATH=$CONTAINER_PATH \\
  --env LD_LIBRARY_PATH=$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-} \\
  --env TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH \\
  --env LIBRARY_PATH=$TRITON_LIBCUDA_PATH:${LIBRARY_PATH:-} \\
  --env PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-} \\
  --env TORCH_HOME=$TORCH_HOME \\
  --env HF_HOME=$HF_HOME \\
  --env PIP_CACHE_DIR=$PIP_CACHE_DIR \\
  --env XDG_CACHE_HOME=$XDG_CACHE_HOME \\
  --env PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX \\
  --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \\
  --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \\
  --env TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR \\
  --env UV_CACHE_DIR=$UV_CACHE_DIR \\
  --env UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR \\
  --env XLA_FLAGS="$XLA_FLAGS" \\
  --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \\
  --env XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION \\
  --env JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR \\
  --env VENV=$VENV \\
  --env SCRATCH=$SCRATCH \\
  "$IMG" \\
  bash -lc "set -euo pipefail; source '$ENV_DIR/bin/activate'; cd '$REPO_DIR'; exec bash '$JOB_ABS'"
EOF
} > "$WRAP"

chmod +x "$WRAP"
echo "[wrapper] Submitting: $WRAP"
sbatch "$WRAP"