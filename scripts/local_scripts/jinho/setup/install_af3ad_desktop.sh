#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Setup Script
# =============================================================================
# This script sets up the complete environment for allatom-design on a GTX1080
# desktop with Ubuntu 22.04. It creates a virtual environment and installs all
# necessary dependencies.
#
# Usage: bash scripts/local_scripts/jinho/setup/install_af3ad_desktop.sh
# =============================================================================

set -e  # Exit on any error

echo "Starting AF3AD desktop environment setup..."

# Configuration
VENV_NAME="af3ad_desktop"
VENV_PATH="$HOME/venv/$VENV_NAME"
UV_CACHE_DIR="/home/possu/jinho/uv/cache"
UV_PYTHON_DIR="/home/possu/jinho/uv/python"
HMMER_BUILD_DIR="/home/possu/jinho/hmmer_build"
HMMER_INSTALL_DIR="/home/possu/jinho/hmmer"

echo "Configuration:"
echo "  Virtual Environment: $VENV_PATH"
echo "  UV Cache Dir: $UV_CACHE_DIR"
echo "  UV Python Dir: $UV_PYTHON_DIR"
echo "  HMMER Build Dir: $HMMER_BUILD_DIR"
echo "  HMMER Install Dir: $HMMER_INSTALL_DIR"
echo ""

# Step 1: Create UV virtual environment
echo "Step 1: Creating UV virtual environment..."
UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR" uv venv "$VENV_PATH" --python 3.12
echo "✓ Virtual environment created"

# Step 2: Install pip in the virtual environment
echo "Step 2: Installing pip..."
source "$VENV_PATH/bin/activate"
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install pip
echo "✓ pip installed"

# Step 3: Install PyTorch
echo "Step 3: Installing PyTorch 2.5.1+cu121..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"

# Step 4: Install JAX
echo "Step 4: Installing JAX 0.4.34+cuda12..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install jax==0.4.34
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --upgrade jax[cuda12]==0.4.34 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
echo "✓ JAX installed"

# Step 5: Install HMMER from source with seq_limit patch
echo "Step 5: Installing HMMER from source..."
mkdir -p "$HMMER_BUILD_DIR" "$HMMER_INSTALL_DIR"
cd "$HMMER_BUILD_DIR"

# Download HMMER
if [ ! -f "hmmer-3.4.tar.gz" ]; then
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
fi

# Verify checksum
echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check

# Extract
if [ ! -d "hmmer-3.4" ]; then
    tar zxf hmmer-3.4.tar.gz
fi

# Apply seq_limit patch
if [ ! -f "jackhmmer_seq_limit.patch" ]; then
    cp "$(dirname "$0")/../../../../alphafold3/docker/jackhmmer_seq_limit.patch" .
fi
cd hmmer-3.4
if [ ! -f ".patch_applied" ]; then
    patch -p0 < ../jackhmmer_seq_limit.patch
    touch .patch_applied
fi

# Build and install HMMER
./configure --prefix "$HMMER_INSTALL_DIR"
make -j$(nproc)
make install
cd easel && make install
echo "✓ HMMER installed"

# Step 6: Return to allatom-design directory and install AlphaFold3
echo "Step 6: Installing AlphaFold3..."
cd "$(dirname "$0")/../../../.."  # Go to allatom-design root
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install setuptools
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -r alphafold3/dev-requirements.txt
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -e ./alphafold3 --no-deps
build_data
echo "✓ AlphaFold3 installed"

# Step 7: Install allatom_design dependencies
echo "Step 7: Installing allatom_design dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install mashumaro==3.16
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torch-cluster==1.6.3+pt25cu121 torch-geometric==2.6.1 --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install nglview==3.1.4
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install lightning hydra-core wandb biotite
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install pandas seaborn matplotlib
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torchtyping einops biopython
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install ihm modelcif
echo "✓ All dependencies installed"

# Step 8: Configure GTX1080 environment variables
echo "Step 8: Configuring GTX1080 environment variables..."
cat >> "$VENV_PATH/bin/activate" << 'EOF'

# GTX1080 AF3 compatibility settings
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export JAX_LOG_COMPILES=1
export TORCH_COMPILE=0
export PATH="/home/possu/jinho/hmmer/bin:$PATH"
echo "GTX1080 AF3 environment loaded"
EOF
echo "✓ Environment variables configured"

# Step 9: Clean up build directory
echo "Step 9: Cleaning up..."
rm -rf "$HMMER_BUILD_DIR"
echo "✓ Cleanup completed"

echo ""
echo "============================================================================="
echo "AF3AD Desktop Environment Setup Complete!"
echo "============================================================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To test the setup, run:"
echo "  python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_seq_denoiser_local.yaml"
echo ""
echo "Environment configured for GTX1080 with:"
echo "  - PyTorch 2.5.1+cu121"
echo "  - JAX 0.4.34+cuda12"
echo "  - HMMER 3.4 with seq_limit patch"
echo "  - AlphaFold3 with no-deps installation"
echo "  - All allatom_design dependencies"
echo "  - GTX1080 compatibility settings"
echo "============================================================================="
