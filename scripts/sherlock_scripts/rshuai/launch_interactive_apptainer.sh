#!/usr/bin/env bash
set -euo pipefail

# ------------------ EDIT THESE ONCE ------------------
IMG="$GROUP_HOME/containers/pytorch_25.08.sif"
REPO_DIR="/home/users/rshuai/code/allatom-design"
ENV_DIR="$SCRATCH/envs"
# ----------------------------------------------------

APPTAINER_BIN="${APPTAINER:-apptainer}"
# Default: use GPU support. Pass --no-nv to disable.
NV_FLAG="--nv"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--no-nv] [extra_bind1,extra_bind2,...]

Launch an interactive apptainer shell with your env auto-activated and repo checked out.

Options:
  --no-nv                  Do NOT pass --nv to apptainer (no GPU support).
  extra binds (optional)   Comma-separated additional paths to bind into container.
EOF
  exit 1
}

# parse args (very small/simple)
EXTRA_BINDS=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    --no-nv) NV_FLAG=""; shift ;;
    -h|--help) usage ;;
  esac
fi
if [[ $# -ge 1 ]]; then
  EXTRA_BINDS="$1"
fi

# sanity checks
if ! command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
  echo "Error: '$APPTAINER_BIN' not found in PATH." >&2
  exit 2
fi
if [[ ! -f "$IMG" ]]; then
  echo "Error: Apptainer image not found: $IMG" >&2
  exit 3
fi

# avoid entering container when already inside one
if [[ -n "${INSIDE_APPTAINER:-}" ]]; then
  echo "You appear to already be inside an apptainer. Exiting." >&2
  exit 0
fi

# Build binds (dedupe lightly)
BIND_LIST="$HOME"
[[ -n "${SCRATCH:-}" ]] && BIND_LIST="$BIND_LIST,$SCRATCH"
[[ -n "${GROUP_HOME:-}" ]] && BIND_LIST="$BIND_LIST,$GROUP_HOME"
BIND_LIST="$BIND_LIST,$REPO_DIR,$ENV_DIR"
if [[ -n "$EXTRA_BINDS" ]]; then
  BIND_LIST="$BIND_LIST,$EXTRA_BINDS"
fi

echo "[launch] image: $IMG"
echo "[launch] binds: $BIND_LIST"
if [[ -n "$NV_FLAG" ]]; then echo "[launch] GPU support: enabled"; else echo "[launch] GPU support: disabled"; fi

# Run interactive shell inside container, pre-activate env and cd to repo
# Use exec bash --noprofile --norc -i so environment is clean/interactive
"$APPTAINER_BIN" exec $NV_FLAG --bind "$BIND_LIST" "$IMG" \
  bash -lc '
    set -euo pipefail
    export INSIDE_APPTAINER=1

    # Activate venv if present
    if [ -f "'"$ENV_DIR"'/allatom_design/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "'"$ENV_DIR"'/allatom_design/bin/activate"
      echo "[inside] activated: '"$ENV_DIR"'/allatom_design"
    else
      echo "[inside] venv not found: '"$ENV_DIR"'/allatom_design (continuing without activate)"
    fi

    # cd to repo if available
    if [ -d "'"$REPO_DIR"'" ]; then
      cd "'"$REPO_DIR"'"
      echo "[inside] cwd: $(pwd)"
    fi

    # --- Override the prompt to something familiar ---
    # simple: user@host:cwd$
    export PS1="\u@\h:\w\$ "

    # If you want a colored prompt, use e.g.:
    # export PS1="\[\e[32m\]\u@\h\[\e[0m\]:\w\$ "

    # Drop to an interactive shell WITHOUT sourcing host dotfiles (clean prompt)
    exec bash --noprofile --norc -i
  '
