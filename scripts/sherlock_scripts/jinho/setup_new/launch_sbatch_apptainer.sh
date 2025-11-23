#!/usr/bin/env bash
set -euo pipefail

# --- EDIT THESE THREE (hardcode your setup) ---
IMG="$GROUP_HOME/containers/pytorch_24.12.sif"
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
SNAP="$WRAP_DIR/${JOB_BASE%.sh}.snapshot.${STAMP}.sh"
cp -a "$JOB_ABS" "$SNAP"
chmod +x "$SNAP"
WRAP="$WRAP_DIR/${JOB_BASE%.sh}.container.${STAMP}.sbatch"

# Simple bind list (add what you need)
BIND="$HOME"
[[ -n "${SCRATCH:-}"     ]] && BIND="$BIND,$SCRATCH"
[[ -n "${GROUP_HOME:-}"  ]] && BIND="$BIND,$GROUP_HOME"
BIND="$BIND,$REPO_DIR,$ENV_DIR,$JOB_DIR,$WRAP_DIR"

{
  echo '#!/usr/bin/env bash'
  # Freeze the Slurm directives from the snapshot
  grep -E '^[[:space:]]*#SBATCH' "$SNAP" || true
  cat <<EOF
set -euo pipefail
echo "[container] image: $IMG"
echo "[container] binds: $BIND"
echo "[container] job snapshot: $SNAP"

$APPTAINER_BIN exec --nv --bind "$BIND" "$IMG" "$SNAP"
EOF
} > "$WRAP"

chmod +x "$WRAP"
sbatch "$WRAP"