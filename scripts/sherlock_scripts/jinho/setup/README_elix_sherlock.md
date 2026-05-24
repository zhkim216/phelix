# Elix Sherlock Environment

This guide describes the validated Elix setup for Sherlock: build the
Apptainer image locally, copy it to Sherlock, install the uv Python environment
inside the container, validate the install, and submit jobs through the wrapper.

Elix uses the AlphaFold3-new tree plus editable `atomworks` and
`allatom-design` checkouts. The previous Sherlock environment is not reused
here.

## 1. Path Variables

Set these variables explicitly before following the commands. Adjust the repo
name if your checkout is not named `elix`.

Local workstation:

```bash
export LOCAL_REPO=/path/to/elix
export LOCAL_WORK=/path/to/elix-build
export LOCAL_SIF="$LOCAL_WORK/elix.sif"
export LOCAL_APPTAINER_TMP="$LOCAL_WORK/apptainer_tmp"
export LOCAL_APPTAINER_CACHE="$LOCAL_WORK/apptainer_cache"
export LOCAL_LOG_DIR="$LOCAL_WORK/logs"
```

Sherlock paths, set from your local workstation before copying the image:

```bash
export SHERLOCK_HOST=sherlock
export SHERLOCK_USER=your_sherlock_username
export SHERLOCK_LOGIN="$SHERLOCK_USER@$SHERLOCK_HOST"
export SHERLOCK_HOME="/home/users/$SHERLOCK_USER"
export SHERLOCK_REPO="$SHERLOCK_HOME/code/elix"
export SHERLOCK_SCRATCH="/scratch/users/$SHERLOCK_USER"
export SHERLOCK_SIF="$SHERLOCK_SCRATCH/containers/elix.sif"
```

Inside a Sherlock shell, the same paths are:

```bash
export PROJECT_ROOT="$HOME/code/elix"
export SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export SIF="$SCRATCH/containers/elix.sif"
export UV_ENV_ROOT="$SCRATCH/envs/uv"
export VENV="$UV_ENV_ROOT/elix"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export UV_PYTHON_INSTALL_DIR="$UV_ENV_ROOT/python"
```

The setup scripts use the same defaults: `SCRATCH=/scratch/users/$USER`,
`SIF=$SCRATCH/containers/elix.sif`, `VENV=$SCRATCH/envs/uv/elix`, and
`PROJECT_ROOT=$HOME/code/elix`.

## 2. Migrate An Existing Sherlock Install

If you already installed this project under a previous environment name, do not
rename the old uv environment in place. Editable install metadata and console
script shebangs can keep stale absolute paths. Move the checkout, copy or
rebuild the SIF, and create a fresh uv environment.

Set the previous basename once, then migrate to `elix`:

```bash
export OLD_NAME=previous_install_name
export NEW_NAME=elix

mv "$HOME/code/$OLD_NAME" "$HOME/code/$NEW_NAME"
cd "$HOME/code/$NEW_NAME"
git pull

mkdir -p "$SCRATCH/containers"
cp "$SCRATCH/containers/$OLD_NAME.sif" "$SCRATCH/containers/$NEW_NAME.sif"

rm -rf "$SCRATCH/envs/uv/$NEW_NAME"
bash scripts/sherlock_scripts/jinho/setup/shell_in_container_elix.sh
```

Inside the container:

```bash
bash scripts/sherlock_scripts/jinho/setup/install_elix_sherlock.sh
```

Copying the previous SIF is enough for a fast migration because the runtime
contents are name-independent. Rebuild with `build_elix_sif.sh` if you also want
the image labels/metadata to use the new name.

Only remove the previous environment and container after the Elix validation
commands below pass:

```bash
rm -rf "$SCRATCH/envs/uv/$OLD_NAME"
rm -f "$SCRATCH/containers/$OLD_NAME.sif"
```

## 3. Build The SIF Locally

The recommended path is to build the image on a local Linux workstation with
Apptainer and then copy the resulting `.sif` to Sherlock. This avoids Sherlock
fakeroot and system-package limitations.

Local preflight:

```bash
apptainer --version
uname -m
command -v mksquashfs
command -v newuidmap || true
command -v newgidmap || true
test -f "$LOCAL_REPO/alphafold3/docker/jackhmmer_seq_limit.patch"
```

Build with real root and explicit cache/temp locations:

```bash
mkdir -p "$LOCAL_APPTAINER_TMP" "$LOCAL_APPTAINER_CACHE" "$LOCAL_LOG_DIR"

cd "$LOCAL_REPO"
BUILD_SCRIPT="$LOCAL_REPO/scripts/sherlock_scripts/jinho/setup/build_elix_sif.sh"

sudo env \
  SIF="$LOCAL_SIF" \
  FORCE=1 \
  APPTAINER_BUILD_FLAGS= \
  APPTAINER_TMPDIR="$LOCAL_APPTAINER_TMP" \
  APPTAINER_CACHEDIR="$LOCAL_APPTAINER_CACHE" \
  bash "$BUILD_SCRIPT" \
  2>&1 | tee "$LOCAL_LOG_DIR/elix_sif_build.log"

sudo chown "$USER:$USER" "$LOCAL_SIF"
```

Validate the local image:

```bash
ls -lh "$LOCAL_SIF"
apptainer exec "$LOCAL_SIF" \
  bash -lc 'python3.12 --version && uv --version && gcc --version | head -1 && g++ --version | head -1 && jackhmmer -h | grep -- --seq_limit'
```

The `--seq_limit` check is important. AlphaFold3's Docker/source tree builds
HMMER 3.4 from source and applies `alphafold3/docker/jackhmmer_seq_limit.patch`;
the AF3 Jackhmmer wrapper uses this option when available to reduce redundant
output and peak memory.

## 4. Copy The SIF To Sherlock

Copy the image into your Sherlock scratch container directory:

```bash
ssh "$SHERLOCK_LOGIN" "mkdir -p '$SHERLOCK_SCRATCH/containers'"
rsync -avP "$LOCAL_SIF" "$SHERLOCK_LOGIN:$SHERLOCK_SIF"
```

On Sherlock, validate the copied image:

```bash
cd "$PROJECT_ROOT"
git pull

apptainer exec --nv "$SIF" \
  bash -lc 'python3.12 --version && uv --version && gcc --version | head -1 && g++ --version | head -1 && jackhmmer -h | grep -- --seq_limit'
```

Expected output should include Python 3.12, uv, GCC/G++, and the
`jackhmmer --seq_limit` help entry.

## 5. Alternative: Build The SIF On Sherlock

Use this only if local building is unavailable. The helper builds from the
NVIDIA CUDA 12.6 Ubuntu 24.04 Docker base and adds Python 3.12, uv, zlib
development headers, and patched HMMER.

```bash
cd "$PROJECT_ROOT"
bash scripts/sherlock_scripts/jinho/setup/build_elix_sif.sh
```

If Sherlock does not allow `--fakeroot`, retry without build flags:

```bash
APPTAINER_BUILD_FLAGS= \
  bash scripts/sherlock_scripts/jinho/setup/build_elix_sif.sh
```

If a partial image already exists:

```bash
FORCE=1 bash scripts/sherlock_scripts/jinho/setup/build_elix_sif.sh
```

## 6. Install The uv Environment On Sherlock

Open a container shell:

```bash
cd "$PROJECT_ROOT"
bash scripts/sherlock_scripts/jinho/setup/shell_in_container_elix.sh
```

For Glide/Schrodinger jobs, opt in explicitly:

```bash
bash scripts/sherlock_scripts/jinho/setup/shell_in_container_elix.sh --schrodinger
```

Inside the container, install Elix:

```bash
rm -rf "$VENV"
bash scripts/sherlock_scripts/jinho/setup/install_elix_sherlock.sh
```

The installer creates `$VENV`, installs the Torch/PyG Sherlock pins, installs
AlphaFold3-new, runs `build_data`, and installs `atomworks` and
`allatom-design` in editable mode. It intentionally does not install the old
`requirements_split/sherlock/*core*.txt` files because those belonged to the old
AlphaFold3 dependency set.

## 7. Validate The Install

Inside the container:

```bash
source "$VENV/bin/activate"

python -c 'import jax, torch, rdkit, alphafold3, atomworks, allatom_design; print("jax", jax.__version__, jax.devices()); print("torch", torch.__version__, torch.cuda.is_available()); print("rdkit", rdkit.__version__); print("alphafold3", alphafold3.__file__); print("atomworks", atomworks.__file__)'
```

Run the tracked install smoke test:

```bash
ELIX_INSTALL_TEST_STRICT=1 python -m pytest tests/test_elix_install.py -q
```

Inside a GPU allocation, also require CUDA/GPU visibility:

```bash
ELIX_INSTALL_TEST_STRICT=1 ELIX_REQUIRE_GPU=1 \
  python -m pytest tests/test_elix_install.py -q
```

Run the lightweight AlphaFold3 data-pipeline smoke tests:

```bash
python alphafold3/run_alphafold_data_test.py \
  DataPipelineTest.test_template_chain_id_roundtrip \
  DataPipelineTest.test_ligand_template_conditioning_config \
  DataPipelineTest.test_ligand_template_conditioning_rejects_zero_templates
```

Known `pip check` metadata conflicts:

- `atomworks` declares `rdkit<2025.9`, while AlphaFold3-new uses
  `rdkit==2025.09.4`.
- `torch==2.7.0+cu126` declares `nvidia-cudnn-cu12==9.5.1.17`, while JAX 0.9.1
  needs newer cuDNN. This setup pins `nvidia-cudnn-cu12==9.22.0.52`.

These conflicts are expected for this environment; validate runtime behavior
with the smoke tests above.

## 8. Submit Jobs

Wrap an existing sbatch script:

```bash
cd "$PROJECT_ROOT"
bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh path/to/job.sbatch
```

For Glide/Schrodinger jobs:

```bash
bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh --schrodinger path/to/job.sbatch
```

Dry-run wrapper generation without submitting:

```bash
bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh --dry-run path/to/job.sbatch
```

The wrapper activates `$VENV` inside `$SIF`, binds the repo and cache
directories, and runs the original sbatch body from `$PROJECT_ROOT`.

## 9. Troubleshooting

- If the container shell says the image is missing, check `echo "$SIF"` and
  confirm that the copied file exists on Sherlock.
- If `jackhmmer -h | grep -- --seq_limit` fails, rebuild the image from a repo
  that contains `alphafold3/docker/jackhmmer_seq_limit.patch`.
- If imports resolve outside the checkout, rerun the installer and check
  `PROJECT_ROOT`.
- If GPU tests fail on a login node, retry inside an interactive GPU allocation.
