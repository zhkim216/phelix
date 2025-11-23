# Lullaby (Ligand-conditioned Caliby)

Branch for developing Lullaby with alphafold3 and openstructure.

Below are the instructions for setting up the codebase, container, and environment on Sherlock.

## Environment Setup

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

### 3. Create Virtual Environment

Create a UV virtual environment in your `$SCRATCH` directory:

```bash
# Create venv directory
mkdir -p $SCRATCH/venv

# Create af3ad virtual environment
cd $SCRATCH/venv
uv venv lullaby --python 3.12

# Activate the environment
source $SCRATCH/venv/lullaby/bin/activate
```

### 4. Container Setup

Copy the container to your `$SCRATCH/containers` directory:

```bash
mkdir -p $SCRATCH/containers
cp /oak/stanford/groups/possu/jinho/containers/lullaby.sif $SCRATCH/containers
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
cd $HOME/code/allatom-design/requirements_split

# Upgrade setuptools and wheel
uv pip install --upgrade setuptools wheel pip

# Install PyTorch dependencies
uv pip install -r uv-compatible-torch.txt --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install -r uv-compatible-core.txt --no-deps

# Install additional dependencies via pip
python -m pip install -r pip-only-torch.txt --no-deps
pip install -r pip-only-core.txt --no-deps

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
cd $HOME/code/allatom-design/atomworks
uv pip install -e . --no-deps

#### Allatom Design

Before installing allatom_design, clean any existing installation (if present):

```bash
# Optional: Remove existing egg-info if present
rm -rf allatom_design.egg-info
pip cache purge
```

Install allatom_design in editable mode:

```bash
cd $HOME/code/allatom-design
pip install -e . --no-deps
```

## Running the Container

To start an interactive shell session in the container:

```bash
cd $HOME/code/allatom-design
bash ./scripts/sherlock_scripts/jinho/shell_in_container.sh
```

## Troubleshooting

- If you encounter issues with package installations, ensure all dependency files are present in the repository
- For container-related issues, verify that the `.sif` file was copied correctly to your `$SCRATCH/containers` directory
- Check that all paths in the setup scripts point to your actual directory locations