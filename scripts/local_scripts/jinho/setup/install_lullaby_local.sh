#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Setup Script (Hybrid: Mamba + UV)
# =============================================================================
# This script sets up the complete environment for allatom-design on a GTX1080
# using a hybrid approach:
# 1. Mamba (Conda) for OpenStructure (complex C++ dependencies)
# 2. UV for everything else (fast, clean Python package management)
#
# Prerequisite:
#   mamba create -n lullaby_local python=3.10 openstructure -c bioconda -c conda-forge -y
#   mamba activate lullaby_local
#
# Usage: bash scripts/local_scripts/jinho/setup/install_lullaby_local.sh
# =============================================================================

set -e  # Exit on any error

# Configuration
ENV_NAME="lullaby_local"
UV_CACHE_DIR="$HOME/jinho/uv/cache"
HMMER_BUILD_DIR="$HOME/jinho/hmmer_build"
HMMER_INSTALL_DIR="$HOME/jinho/hmmer"

# Check if correct conda environment is active
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Error: '$ENV_NAME' Conda environment is not active."
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: mamba activate $ENV_NAME"
    exit 1
fi

echo "Starting AF3AD desktop environment setup (Hybrid Mode)..."
echo "Configuration:"
echo "  Active Environment: $CONDA_DEFAULT_ENV"
echo "  Python Location: $(which python)"
echo "  HMMER Install Dir: $HMMER_INSTALL_DIR"
echo ""

# Step 1: Install pip & build tools via UV
echo "Step 1: Updating pip and build tools..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --upgrade pip setuptools wheel
# Install exceptiongroup for Python 3.10 compatibility (needed for atomworks)
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install exceptiongroup
echo "✓ Basic tools updated (including exceptiongroup)"

# Step 2: Install PyTorch (GTX1080 compatible)
echo "Step 2: Installing PyTorch 2.5.1+cu121..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"

# Step 3: Install JAX
echo "Step 3: Installing JAX 0.4.34+cuda12..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install jax==0.4.34
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --upgrade jax[cuda12]==0.4.34 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo "✓ JAX installed"

# Step 4: Install HMMER from source with seq_limit patch
echo "Step 4: Installing HMMER from source..."
mkdir -p "$HMMER_BUILD_DIR" "$HMMER_INSTALL_DIR"
cd "$HMMER_BUILD_DIR"

if [ ! -f "hmmer-3.4.tar.gz" ]; then
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
fi

# Verify checksum
echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check

if [ ! -d "hmmer-3.4" ]; then
    tar zxf hmmer-3.4.tar.gz
fi

# Apply seq_limit patch logic
PATCH_PATH="$(dirname "$0")/../../../../alphafold3/docker/jackhmmer_seq_limit.patch"
# Fallback to absolute path search if relative fails
if [ ! -f "$PATCH_PATH" ]; then
    PATCH_PATH="$HOME/jinho/allatom-design/alphafold3/docker/jackhmmer_seq_limit.patch"
fi

if [ -f "$PATCH_PATH" ]; then
    cp "$PATCH_PATH" .
    cd hmmer-3.4
    if [ ! -f ".patch_applied" ]; then
        echo "Applying jackhmmer_seq_limit.patch..."
        patch -p0 < ../jackhmmer_seq_limit.patch
        touch .patch_applied
    fi
else
    echo "Warning: jackhmmer_seq_limit.patch not found! Skipping patch."
    cd hmmer-3.4
fi

./configure --prefix "$HMMER_INSTALL_DIR"
make -j$(nproc)
make install
cd easel && make install
echo "✓ HMMER installed"

# Step 5: Install AlphaFold3
echo "Step 5: Installing AlphaFold3..."
# Go back to allatom-design root
cd "$HOME/jinho/allatom-design"

UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -r alphafold3/dev-requirements.txt
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -e ./alphafold3 --no-deps
if command -v build_data &> /dev/null; then
    build_data
fi
echo "✓ AlphaFold3 installed"

# Step 6: Install allatom_design dependencies
echo "Step 6: Installing allatom_design dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install mashumaro==3.16
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torch-cluster==1.6.3+pt25cu121 torch-geometric==2.6.1 --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install nglview==3.1.4
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install lightning hydra-core wandb biotite
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install pandas seaborn matplotlib
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torchtyping einops biopython
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install ihm modelcif
echo "✓ All dependencies installed"

# Step 7: Atomworks & Utils
echo "Step 7: Installing atomworks dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
"biotite>=1.3.0,<2" \
"hydride>=1.2.3,<2" \
"py3Dmol>=2.2.1,<3" \
"pymol-remote>=0.0.5" \
"pyarrow==17.0.0" \
"cython>=3,<4" \
"cytoolz>=0.12.3,<1" \
"typer>=0.12.5,<1"
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install "openbabel-wheel==3.1.1.22" pathspec

echo "Step 7.2: Installing atomworks..."
if [ -d "atomworks" ]; then
    cd atomworks
    # Note: We already patched __init__.py and pyproject.toml for Python 3.10 support
    UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -e . --no-deps
    cd ..
fi
echo "✓ atomworks installed"

# Step 8: Cleanup
echo "Step 8: Cleaning up..."
rm -rf "$HMMER_BUILD_DIR"

echo ""
echo "============================================================================="
echo "AF3AD Desktop Environment Setup Complete!"
echo "============================================================================="
echo "Environment: $ENV_NAME"
echo "Python Version: $(python --version)"
echo "OpenStructure: $(python -c 'import ost; print("Installed")' 2>/dev/null || echo "Not Found")"
echo ""
echo "To activate environment variables for GTX1080:"
echo "  source scripts/local_scripts/jinho/setup/activate_lullaby_local.sh"
echo ""
