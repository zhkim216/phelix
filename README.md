# Ligand-Conditioned Potts Models

_Note: this README was LLM-written and manually reviewed._

This repository contains ligand-conditioned Potts/model development code around
AlphaFold3-new, AtomWorks, and `allatom_design`. The active environment is named
Elix.

## Repository Layout

- `allatom_design/`: main Python package for data, model, evaluation, and
  utility code.
- `alphafold3/`: editable AlphaFold3-new checkout used by Elix.
- `atomworks/`: editable AtomWorks checkout used by Elix.
- `requirements_split/`: uv/pip requirement splits for local and Sherlock
  installs.
- `scripts/`: local and Sherlock setup, training, evaluation, and preprocessing
  scripts.
- `tests/`: lightweight pytest coverage, including Elix install smoke tests.

## Sherlock Setup

Use the Elix-specific Sherlock guide:

```text
scripts/sherlock_scripts/jinho/setup/README_elix_sherlock.md
```

That guide covers the full flow:

1. Build `elix.sif` locally with Apptainer.
2. Copy the SIF to Sherlock.
3. Install the Sherlock uv environment under scratch.
4. Validate imports, GPU visibility, patched HMMER `--seq_limit`, and AF3 data
   smoke tests.
5. Submit jobs through `wrap_sbatch_in_container_elix.sh`.

The previous Sherlock setup is intentionally not documented here because it used
a different AlphaFold3 dependency set.

## Local Desktop Setup

The local desktop environment uses a uv-managed Python 3.12 virtual environment
for the AlphaFold3-new tree. Micromamba is used only to provide zlib headers and
libraries while building AlphaFold3. OpenStructure is not installed.

### 1. Paths And Helper Tools

```bash
export MODEL_DEV=/path/to/model-dev
export REPO="$MODEL_DEV/allatom-design"
export VENV="$MODEL_DEV/envs/uv/elix_local"
export UV_CACHE_DIR="$MODEL_DEV/cache/uv"
export BUILD_PREFIX="$MODEL_DEV/envs/micromamba/envs/af3_build_deps"
export MAMBA_ROOT_PREFIX="$MODEL_DEV/envs/micromamba"

# Use the uv binary on PATH, or set this to a known uv executable.
export UV_BIN=uv
```

Install zlib into the small build prefix:

```bash
micromamba create -y -p "$BUILD_PREFIX" -c conda-forge zlib
```

### 2. Create The uv Environment

```bash
mkdir -p "$(dirname "$VENV")" "$UV_CACHE_DIR"
"$UV_BIN" python install 3.12
"$UV_BIN" venv "$VENV" -p 3.12
"$UV_BIN" pip install --python "$VENV/bin/python" --upgrade pip setuptools wheel
```

### 3. Install PyTorch/PyG And Build Tools

```bash
cd "$REPO"

"$UV_BIN" pip install --python "$VENV/bin/python" \
  scikit_build_core pybind11 'cmake>=3.28' ninja setuptools_scm

"$UV_BIN" pip install --python "$VENV/bin/python" \
  -r requirements_split/local/uv-compatible-torch.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

"$UV_BIN" pip install --python "$VENV/bin/python" \
  -r requirements_split/local/pip-only-torch.txt \
  --index-strategy unsafe-best-match
```

### 4. Install AlphaFold3

```bash
cd "$REPO"

CMAKE_PREFIX_PATH="$BUILD_PREFIX" ZLIB_ROOT="$BUILD_PREFIX" \
  "$UV_BIN" pip install --python "$VENV/bin/python" \
  -e ./alphafold3 --no-build-isolation --index-strategy unsafe-best-match

"$VENV/bin/build_data"
```

### 5. Install AtomWorks And allatom-design

AtomWorks currently declares `rdkit<2025.9`, while AlphaFold3-new requires
`rdkit==2025.09.4`. Keep the AlphaFold3-new dependency set and install
AtomWorks with `--no-deps`.

```bash
cd "$REPO"

"$UV_BIN" pip install --python "$VENV/bin/python" \
  mashumaro==3.14 nglview==3.1.4 hydra-core wandb \
  'pandas>=2.2,<2.4' seaborn matplotlib torchtyping einops biopython ihm modelcif \
  biotite==1.2.0 hydride 'py3Dmol>=2.2.1,<3' 'pymol-remote>=0.0.5' \
  pyarrow==17.0.0 'cython>=3,<4' 'cytoolz>=0.12.3,<1' 'typer>=0.12.5,<1' \
  'jaxtyping>=0.2.17,<1' 'beartype>=0.18.0,<1' pathspec pytest \
  --index-strategy unsafe-best-match

"$UV_BIN" pip install --python "$VENV/bin/python" openbabel-wheel==3.1.1.22
"$UV_BIN" pip install --python "$VENV/bin/python" -e ./atomworks --no-deps
"$UV_BIN" pip install --python "$VENV/bin/python" -e . --no-deps
```

JAX 0.9.1 needs a newer cuDNN runtime than the exact version declared by
`torch==2.7.0+cu126`. Keep the newer cuDNN package in the environment:

```bash
"$UV_BIN" pip install --python "$VENV/bin/python" \
  nvidia-cudnn-cu12==9.22.0.52 --index-strategy unsafe-best-match
```

### 6. Use The Environment

```bash
source "$VENV/bin/activate"
cd "$REPO"
```

For local GPUs with compute capability 7.x, use XLA flash attention plus the
XLA compatibility flag:

```bash
export XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter

python alphafold3/run_alphafold.py \
  --flash_attention_implementation=xla \
  ...
```

### 7. Validate

```bash
python -m compileall -q alphafold3/src/alphafold3 atomworks/src/atomworks allatom_design

python -c 'import jax, torch, rdkit, alphafold3, atomworks, allatom_design; print("jax", jax.__version__, jax.devices()); print("torch", torch.__version__, torch.cuda.is_available()); print("rdkit", rdkit.__version__); print("alphafold3", alphafold3.__file__); print("atomworks", atomworks.__file__)'

python alphafold3/run_alphafold_data_test.py \
  DataPipelineTest.test_template_chain_id_roundtrip \
  DataPipelineTest.test_ligand_template_conditioning_config \
  DataPipelineTest.test_ligand_template_conditioning_rejects_zero_templates
```

Expected `pip check` metadata conflicts:

- `atomworks` declares `rdkit<2025.9`, while AlphaFold3-new uses
  `rdkit==2025.09.4`.
- `torch==2.7.0+cu126` declares `nvidia-cudnn-cu12==9.5.1.17`, while JAX 0.9.1
  inference uses `nvidia-cudnn-cu12==9.22.0.52`.
