#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE THREE (cluster-specific) ---
IMG="${SIF:-/scratch/users/zhkim216/containers/af3ad_base.sif}"       # 컨테이너 이미지(.sif)
REPO_DIR="/home/users/zhkim216/code/allatom-design"                   # 저장소 루트
ENV_DIR="${VENV:-/scratch/users/zhkim216/venv/af3ad}"                 # venv 디렉토리
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

# Binds (최소 필요 경로만)
BIND="$HOME"
[[ -n "${SCRATCH:-}"    ]] && BIND="$BIND,$SCRATCH"
[[ -n "${GROUP_HOME:-}" ]] && BIND="$BIND,$GROUP_HOME"
BIND="$BIND,$REPO_DIR,$ENV_DIR,$JOB_DIR"

{
  echo '#!/usr/bin/env bash'
  # Keep original #SBATCH
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  cat <<EOF
set -euo pipefail
echo "[container] image: $IMG"
echo "[container] binds: $BIND"

$APPTAINER_BIN exec --nv --bind "$BIND" "$IMG" \\
  bash -lc "set -euo pipefail; source '$ENV_DIR/bin/activate'; cd '$REPO_DIR'; exec bash '$JOB_ABS'"
EOF
} > "$WRAP"

chmod +x "$WRAP"
echo "[wrapper] Submitting: $WRAP"
sbatch "$WRAP"