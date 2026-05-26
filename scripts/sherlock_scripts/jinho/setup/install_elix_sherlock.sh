#!/usr/bin/env bash
set -euo pipefail

# Run this inside the Elix Sherlock container.

ELIX_USER="${USER:-$(id -un)}"
ELIX_HOME="${HOME:-/home/users/$ELIX_USER}"
PROJECT_ROOT="${PROJECT_ROOT:-$ELIX_HOME/code/elix}"
SCRATCH="${SCRATCH:-/scratch/users/$ELIX_USER}"
UV_ENV_ROOT="${UV_ENV_ROOT:-$SCRATCH/envs/uv}"
VENV="${VENV:-$UV_ENV_ROOT/elix}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$UV_ENV_ROOT/python}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

export SCRATCH UV_ENV_ROOT UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

need_cmd "$PYTHON_BIN"
need_cmd gcc
need_cmd g++

zlib_header_found=0
for header in /usr/include/zlib.h /usr/local/include/zlib.h /opt/conda/include/zlib.h; do
  if [ -f "$header" ]; then
    zlib_header_found=1
    break
  fi
done
if [ "$zlib_header_found" -ne 1 ]; then
  fail "zlib.h not found. Use an Elix/AF3 container with zlib development headers."
fi

if command -v jackhmmer >/dev/null 2>&1; then
  if ! jackhmmer -h 2>&1 | grep -q -- "--seq_limit"; then
    echo "WARNING: jackhmmer exists but --seq_limit was not found in help output." >&2
  fi
else
  echo "WARNING: jackhmmer not found. AF3 data pipeline will need patched HMMER in PATH." >&2
fi

mkdir -p "$UV_ENV_ROOT" "$(dirname "$VENV")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if ! command -v uv >/dev/null 2>&1; then
  need_cmd curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
need_cmd uv

cd "$PROJECT_ROOT"

uv venv "$VENV" -p "$PYTHON_BIN"
"$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true

uv pip install --python "$VENV/bin/python" --upgrade pip setuptools wheel
uv pip install --python "$VENV/bin/python" \
  scikit_build_core pybind11 'cmake>=3.28' ninja setuptools_scm

uv pip install --python "$VENV/bin/python" \
  -r requirements_split/sherlock/uv-compatible-torch.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

uv pip install --python "$VENV/bin/python" \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html \
  torch_cluster==1.6.3+pt27cu126 \
  torch_scatter==2.1.2+pt27cu126 \
  torch_sparse==0.6.18+pt27cu126 \
  torch_spline_conv==1.2.2+pt27cu126 \
  lightning-utilities==0.11.8 \
  torchmetrics==1.0.3 \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

uv pip install --python "$VENV/bin/python" \
  -e ./alphafold3 --no-build-isolation --index-strategy unsafe-best-match

"$VENV/bin/build_data"

uv pip install --python "$VENV/bin/python" \
  mashumaro==3.14 nglview==3.1.4 hydra-core wandb \
  'pandas>=2.2,<2.4' seaborn matplotlib torchtyping einops biopython ihm modelcif \
  biotite==1.2.0 hydride 'py3Dmol>=2.2.1,<3' 'pymol-remote>=0.0.5' \
  pyarrow==17.0.0 'cython>=3,<4' 'cytoolz>=0.12.3,<1' 'typer>=0.12.5,<1' \
  'jaxtyping>=0.2.17,<1' 'beartype>=0.18.0,<1' pathspec pytest \
  --index-strategy unsafe-best-match

uv pip install --python "$VENV/bin/python" openbabel-wheel==3.1.1.22
uv pip install --python "$VENV/bin/python" -e ./atomworks --no-deps
uv pip install --python "$VENV/bin/python" -e . --no-deps

# JAX 0.9.1 requires newer cuDNN than the exact torch 2.7.0 metadata pin.
uv pip install --python "$VENV/bin/python" \
  nvidia-cudnn-cu12==9.22.0.52 --index-strategy unsafe-best-match

"$VENV/bin/python" -m compileall -q alphafold3/src/alphafold3 atomworks/src/atomworks allatom_design

echo
echo "Elix install complete."
echo "Activate with:"
echo "  source $VENV/bin/activate"
echo
echo "Recommended Sherlock validation:"
cat <<'EOF'
python - <<'PY'
import jax, torch, rdkit, alphafold3, atomworks, allatom_design
print("jax", jax.__version__, jax.devices())
print("torch", torch.__version__, torch.cuda.is_available())
print("rdkit", rdkit.__version__)
print("alphafold3", alphafold3.__file__)
print("atomworks", atomworks.__file__)
PY

ELIX_INSTALL_TEST_STRICT=1 python -m pytest tests/test_elix_install.py -q

python -m pytest alphafold3/tests/test_runner_module.py -q
EOF
echo
echo "pip check may report expected metadata conflicts:"
echo "  atomworks rdkit<2025.9 vs AF3 rdkit==2025.9.4"
echo "  torch cuDNN==9.5.1.17 metadata vs JAX-compatible cuDNN==9.22.0.52"
