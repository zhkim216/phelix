#!/usr/bin/env bash
set -euo pipefail

# Build the Phelix Apptainer image on Sherlock.
# Run from any directory after cloning/pulling the phelix repo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [ -z "${APPTAINER_BIN:-}" ]; then
  if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_BIN="apptainer"
  elif [ -x /bin/singularity ]; then
    APPTAINER_BIN="/bin/singularity"
  else
    APPTAINER_BIN="singularity"
  fi
fi
SIF="${SIF:-/scratch/users/zhkim216/containers/phelix.sif}"
DEF="${DEF:-$SCRIPT_DIR/phelix_apptainer.def}"
APPTAINER_BUILD_FLAGS="${APPTAINER_BUILD_FLAGS:---fakeroot}"
FORCE="${FORCE:-0}"

if ! command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
  echo "ERROR: Apptainer/Singularity command not found: $APPTAINER_BIN" >&2
  exit 1
fi

if [ ! -f "$DEF" ]; then
  echo "ERROR: Apptainer definition file not found: $DEF" >&2
  exit 1
fi

if [ -f "$SIF" ]; then
  if [ "$FORCE" != "1" ]; then
    echo "SIF already exists: $SIF"
    echo "Set FORCE=1 to rebuild it."
    exit 0
  fi
  rm -f "$SIF"
fi

mkdir -p "$(dirname "$SIF")"

cd "$REPO_ROOT"

echo "Building Phelix SIF"
echo "  repo: $REPO_ROOT"
echo "  def:  $DEF"
echo "  sif:  $SIF"
echo "  bin:  $APPTAINER_BIN"
echo "  flags: ${APPTAINER_BUILD_FLAGS:-<none>}"

# shellcheck disable=SC2086
"$APPTAINER_BIN" build $APPTAINER_BUILD_FLAGS "$SIF" "$DEF"

echo
echo "Built: $SIF"
echo "Quick check:"
echo "  $APPTAINER_BIN exec $SIF bash -lc 'python3.12 --version && uv --version && jackhmmer -h | grep -- --seq_limit'"
