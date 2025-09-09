#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE THREE (hardcode your setup) ---
IMG="$GROUP_HOME/containers/pytorch_25.08.sif"
REPO_DIR="/home/users/rshuai/code/allatom-design"
ENV_DIR="$SCRATCH/envs"
# ----------------------------------------------

APPTAINER_BIN="apptainer"

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <sbatch_script.sh>" >&2
  exit 1
fi

JOB="$1"
[[ -f "$JOB" ]] || { echo "Not found: $JOB" >&2; exit 1; }

# Resolve absolute path (portable)
JOB_DIR="$(cd "$(dirname "$JOB")" && pwd)"
JOB_BASE="$(basename "$JOB")"
JOB_ABS="$JOB_DIR/$JOB_BASE"

WRAP_DIR="${SCRATCH:-/tmp}/slurm_apptainer_wrappers"
mkdir -p "$WRAP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP="$WRAP_DIR/${JOB_BASE%.sh}.container.${STAMP}.sbatch"

# Simple bind list (add what you need)
BIND="$HOME"
[[ -n "${SCRATCH:-}"     ]] && BIND="$BIND,$SCRATCH"
[[ -n "${GROUP_HOME:-}"  ]] && BIND="$BIND,$GROUP_HOME"
BIND="$BIND,$REPO_DIR,$ENV_DIR,$JOB_DIR"

{
  echo '#!/usr/bin/env bash'
  # Keep the original Slurm directives so Slurm honors them
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  cat <<EOF
set -euo pipefail
echo "[container] image: $IMG"
echo "[container] binds: $BIND"

$APPTAINER_BIN exec --nv --bind "$BIND" "$IMG" \
  bash -lc "set -euo pipefail; source '$ENV_DIR/allatom_design/bin/activate'; exec bash '$JOB_ABS'"
EOF
} > "$WRAP"

chmod +x "$WRAP"
sbatch "$WRAP"
