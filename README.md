# Ligand-Conditioned Potts Models

_Note: parts of this README were LLM-written and manually reviewed._

## Environment setup on Sherlock

### 1. Clone Repository

Clone the repository and checkout the correct branch:

```bash
# Navigate to home directory
mkdir -p $HOME/code
cd $HOME/code

# Clone the repository
git clone https://github.com/ProteinDesignLab/allatom-design.git

# Navigate to the project directory
cd allatom-design

# Checkout the jinho/af3ppg branch
git checkout jinho/AAA
```

### 2. Install UV Package Manager

Install UV if not already installed:

```bash
# Install UV using curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Verify installation
uv --version
```

### 3. Container Setup

Copy the container to your `$SCRATCH/containers` directory:

```bash
mkdir -p $SCRATCH/containers
cp /oak/stanford/groups/possu/jinho/containers/lullaby.sif $SCRATCH/containers
```

### 4. Create Virtual Environment

```bash
cd $HOME/code/allatom-design
bash scripts/sherlock_scripts/jinho/setup/shell_in_container.sh
```

Create a UV virtual environment in your `$SCRATCH` directory:

```bash
# Create venv directory
mkdir -p $SCRATCH/venv

# Create lullaby virtual environment inheriting the mamba environment of the container
cd $SCRATCH/venv
uv venv lullaby --python=/opt/conda/envs/lullaby/bin/python --system-site-packages

# Activate the environment
source $SCRATCH/venv/lullaby/bin/activate
```

### 5. Update Environment Paths

Modify the paths in `scripts/sherlock_scripts/jinho/setup/env_setup.sh` to match your directory structure.

### 6. Update Script References

In the following scripts:
- `scripts/sherlock_scripts/jinho/setup/run_debugpy_sherlock.sh`
- `scripts/sherlock_scripts/jinho/setup/run_in_container.sh`
- `scripts/sherlock_scripts/jinho/setup/shell_in_container.sh`

Replace:
```bash
source "/home/users/zhkim216/code/allatom-design/scripts/sherlock_scripts/jinho/setup/env_setup.sh"
```

With:
```bash
source "{YOUR_allatom-design-absolute-DIR}/scripts/sherlock_scripts/jinho/env_setup.sh"
```

### 7. Install Dependencies

Start the container and install required packages:

```bash
cd $HOME/code/allatom-design
bash ./scripts/sherlock_scripts/jinho/shell_in_container.sh
```

Inside the container shell:

```bash
cd $HOME/code/allatom-design/requirements_split/sherlock

# Upgrade setuptools and wheel
uv pip install --upgrade setuptools wheel pip

# Install PyTorch dependencies
uv pip install -r uv-compatible-torch.txt --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install -r uv-compatible-core.txt --no-deps

# Install additional dependencies via pip
python -m pip install -r pip-only-torch.txt --no-deps
pip install -r pip-only-core.txt --no-deps

# Install lightning and downgrade networkx to avoid conflicts in openstructure
uv pip install lightning --no-deps
uv pip install networkx==2.8.8

# Install python-dotenv
uv pip install python-dotenv

# Atomworks dependencies (Todo: Integrate these into requirements.txt)
uv pip install \
"biotite>=1.3.0,<2" \
"hydride>=1.2.3,<2" \
"py3Dmol>=2.2.1,<3" \
"pymol-remote>=0.0.5" \
"pyarrow==17.0.0" \
"cython>=3,<4" \
"cytoolz>=0.12.3,<1" \
"typer>=0.12.5,<1"

uv pip install "openbabel-wheel==3.1.1.22"
uv pip install pathspec
```

### 8. Install Editable Packages

#### AlphaFold3

Install AlphaFold3 in editable mode:

```bash
cd $HOME/code/allatom-design/alphafold3
pip install -e . --no-deps
```

#### Atomworks
```bash
cd $HOME/code/allatom-design/atomworks
uv pip install -e . --no-deps
```

#### Allatom Design

Before installing allatom_design, clean any existing installation (if present):

```bash
# Optional: Remove existing egg-info if present
cd $HOME/code/allatom-design
rm -rf allatom_design.egg-info
pip cache purge
```

Install allatom_design in editable mode:

```bash
cd $HOME/code/allatom-design
pip install -e . --no-deps
```

### Running the Container

To start an interactive shell session in the container:

```bash
cd $HOME/code/allatom-design
bash ./scripts/sherlock_scripts/jinho/shell_in_container.sh
```

### Troubleshooting

- If you encounter issues with package installations, ensure all dependency files are present in the repository
- For container-related issues, verify that the `.sif` file was copied correctly to your `$SCRATCH/containers` directory
- Check that all paths in the setup scripts point to your actual directory locations


## Environment setup on the lab desktop

The local desktop environment uses a uv-managed Python 3.12 virtual environment
for the AlphaFold3-new based tree. It keeps the Python environment under
`/home/yjhk/model-dev/envs/uv`, uses micromamba only for the small zlib build
prefix, and does not install OpenStructure. Sherlock/container setup should be
documented separately.

### 1. Paths and helper tools

```bash
export REPO=/home/yjhk/model-dev/allatom-design
export VENV=/home/yjhk/model-dev/envs/uv/phelix_local
export UV_CACHE_DIR=/home/yjhk/model-dev/cache/uv
export BUILD_PREFIX=/home/yjhk/model-dev/envs/micromamba/envs/af3_build_deps
export MAMBA_ROOT_PREFIX=/home/yjhk/model-dev/envs/micromamba

# This was the uv binary available on the desktop when the environment was
# created. If uv is already on PATH, `export UV_BIN=uv` is also fine.
export UV_BIN=/home/yjhk/model-dev/envs/micromamba/envs/phelix_local/bin/uv
```

This setup uses micromamba only to provide zlib headers and libraries while
building AlphaFold3:

```bash
micromamba create -y -p "$BUILD_PREFIX" -c conda-forge zlib
```

### 2. Create the uv environment

```bash
mkdir -p /home/yjhk/model-dev/envs/uv "$UV_CACHE_DIR"
"$UV_BIN" python install 3.12
"$UV_BIN" venv "$VENV" -p 3.12
"$UV_BIN" pip install --python "$VENV/bin/python" --upgrade pip setuptools wheel
```

### 3. Install PyTorch/PyG and build tools

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

### 5. Install allatom-design and AtomWorks

AtomWorks currently declares `rdkit<2025.9`, while AlphaFold3-new requires
`rdkit==2025.9.4`. To keep the AlphaFold3-new dependency set, install the
runtime dependencies explicitly and install AtomWorks with `--no-deps`.

```bash
cd "$REPO"

"$UV_BIN" pip install --python "$VENV/bin/python" \
  mashumaro==3.14 nglview==3.1.4 hydra-core wandb \
  'pandas>=2.2,<2.4' seaborn matplotlib torchtyping einops biopython ihm modelcif \
  biotite==1.2.0 hydride 'py3Dmol>=2.2.1,<3' 'pymol-remote>=0.0.5' \
  pyarrow==17.0.0 'cython>=3,<4' 'cytoolz>=0.12.3,<1' 'typer>=0.12.5,<1' \
  'jaxtyping>=0.2.17,<1' 'beartype>=0.18.0,<1' pathspec \
  --index-strategy unsafe-best-match

"$UV_BIN" pip install --python "$VENV/bin/python" openbabel-wheel==3.1.1.22
"$UV_BIN" pip install --python "$VENV/bin/python" -e ./atomworks --no-deps
"$UV_BIN" pip install --python "$VENV/bin/python" -e . --no-deps
```

JAX 0.9.1 needs a newer cuDNN runtime than the exact version declared by
`torch==2.7.0+cu126`. Keep the newer cuDNN package in the environment because
that is the version validated for local AlphaFold3 inference:

```bash
"$UV_BIN" pip install --python "$VENV/bin/python" \
  nvidia-cudnn-cu12==9.22.0.52 --index-strategy unsafe-best-match
```

### 6. Use the environment

```bash
source /home/yjhk/model-dev/envs/uv/phelix_local/bin/activate

cd /home/yjhk/model-dev/allatom-design
```

The lab desktop currently uses an RTX 2080 Ti. Local AlphaFold3 inference should
use XLA flash attention plus the XLA compatibility flag for compute capability
7.x GPUs:

```bash
export XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter

python alphafold3/run_alphafold.py \
  --flash_attention_implementation=xla \
  ...
```

### 7. Validation notes

The local environment was validated with:

```bash
python -m compileall -q alphafold3/src/alphafold3 atomworks/src/atomworks allatom_design
python alphafold3/run_alphafold_data_test.py \
  DataPipelineTest.test_template_chain_id_roundtrip \
  DataPipelineTest.test_ligand_template_conditioning_config \
  DataPipelineTest.test_ligand_template_conditioning_rejects_zero_templates
```

JAX sees the local GPU and Torch CUDA import works. `python -m pip check` has two
expected metadata conflicts:

- `atomworks 2.2.0` declares `rdkit<2025.9`, while AlphaFold3-new uses
  `rdkit==2025.9.4`.
- `torch 2.7.0+cu126` declares `nvidia-cudnn-cu12==9.5.1.17`, while JAX 0.9.1
  inference uses `nvidia-cudnn-cu12==9.22.0.52`.

The full implementation log and validation output are in
`debug/260523_af3_new_uv_migration/implementation_report.md` and
`debug/260523_af3_new_uv_migration_snapshot/`.
