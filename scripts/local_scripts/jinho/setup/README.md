# AF3AD Desktop Setup

This directory contains setup scripts for running allatom-design on Ubuntu 22.04 desktop environment with GTX1080.

## Files Description

### `install_af3ad_desktop.sh`
Complete environment installation script that performs:
- Creates Python 3.12 virtual environment named `af3ad_desktop`
- Installs PyTorch 2.5.1+cu121
- Installs JAX 0.4.34+cuda12
- Builds HMMER 3.4 from source (with seq_limit patch)
- Installs AlphaFold3 (editable mode, no-deps)
- Installs allatom_design dependencies
- Configures GTX1080 compatibility environment variables

### `activate_af3ad_desktop.sh`
Environment activation script that activates the installed virtual environment.

### `setup_gtx1080_env.sh`
Script for setting GTX1080 compatibility environment variables.

## Usage

### 1. Initial Installation
```bash
# Run from allatom-design root directory
bash scripts/local_scripts/jinho/setup/install_af3ad_desktop.sh
```

### 2. Environment Activation
```bash
# Method 1: Using script
source scripts/local_scripts/jinho/setup/activate_af3ad_desktop.sh

# Method 2: Direct activation (automatically loads GTX1080 settings)
source ~/venv/af3ad_desktop/bin/activate
```

### 3. Test Run
```bash
python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_seq_denoiser_local.yaml
```

## Environment Configuration

### Python Packages
- **PyTorch**: 2.5.1+cu121
- **JAX**: 0.4.34+cuda12
- **Lightning**: Latest
- **AlphaFold3**: Installed in editable mode
- **Allatom-design**: core dependencies (mashumaro, torch-cluster, torch-geometric, nglview, pandas, seaborn, matplotlib, torchtyping, einops, biopython, ihm, modelcif)
- **Extras**: biotite, hydride, py3Dmol, pymol-remote, pyarrow==17.0.0, cython, cytoolz, typer, openbabel-wheel, pathspec

### System Tools
- **HMMER**: 3.4 (with seq_limit patch applied)
- **Installation Path**: `/home/possu/jinho/hmmer/`

### GTX1080 Compatibility Settings
The following environment variables are automatically configured:
```bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export JAX_LOG_COMPILES=1
export TORCH_COMPILE=0
export PATH="/home/possu/jinho/hmmer/bin:$PATH"
```

## Atomworks Installation

The script `install_af3ad_desktop.sh` installs the `atomworks` package from the `atomworks` directory in editable mode (`-e .`) while skipping dependencies (`--no-deps`). This allows source changes to take effect immediately during development.

## Key Changes Made

### Boltz Dependency Removal
Instead of installing the `boltz` package directly, necessary utility functions were copied to `allatom_design/model/seq_denoiser/utils.py`:
- `LinearNoBias`
- `SwiGLU` 
- `center_random_augmentation`
- `randomly_rotate`
- `random_rotations`
- Other geometry functions

### Import Path Updates
- `allatom_design/model/seq_denoiser/denoisers/seq_design/atom_mpnn.py`
- `allatom_design/data/feature/seq_des_featurizer.py`
- `allatom_design/data/feature/motif_featurizer.py`

## Troubleshooting

### CUDA-related Errors
GTX1080 uses Pascal architecture (Compute Capability 6.1), which may have compatibility issues with modern CUDA features. The environment variables resolve these issues.

### Memory Issues
`XLA_CLIENT_MEM_FRACTION=0.8` limits GPU memory usage, and `TF_FORCE_UNIFIED_MEMORY=true` enables unified memory.

### DataLoader Worker Errors
Adjust `num_workers` in config files or reduce `batch_size` if needed.

## Directory Structure
```
/home/possu/jinho/
├── uv/
│   ├── cache/          # UV cache directory  
│   └── python/         # UV Python installations
├── hmmer/              # HMMER installation
│   └── bin/            # HMMER binaries
└── allatom-design/     # Main codebase
```

```
~/venv/
└── af3ad_desktop/      # Virtual environment
    └── bin/activate    # Contains GTX1080 settings
```
