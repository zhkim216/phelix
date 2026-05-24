# Phelix Sherlock Environment

This guide is for the AlphaFold3-new Phelix environment on Sherlock. The old
`lcaliby`/`lullaby` Sherlock environment is intentionally left separate.

## What Codex Prepared Locally

The repo contains Phelix-specific Sherlock helpers:

- `env_setup_phelix.sh`: exports Phelix paths, CUDA/cache variables, and XLA flags.
- `shell_in_container_phelix.sh`: opens an interactive container shell.
- `wrap_sbatch_in_container_phelix.sh`: wraps sbatch jobs in the Phelix container.
- `install_phelix_sherlock.sh`: installs the uv Python environment inside the container.

Defaults:

```bash
SIF=/scratch/users/zhkim216/containers/phelix.sif
VENV=/scratch/users/zhkim216/venv/phelix
PROJECT_ROOT=/home/users/zhkim216/code/phelix
UV_CACHE_DIR=/scratch/users/zhkim216/uv/cache
```

## What Must Be Done On Sherlock

Prepare a container image at:

```bash
/scratch/users/zhkim216/containers/phelix.sif
```

The image must provide:

- Python 3.12
- `gcc`, `g++`, `make`
- zlib development headers (`zlib.h`)
- CUDA 12-compatible runtime
- patched HMMER in `PATH`, ideally under `/hmmer/bin`, with `jackhmmer --seq_limit`

## Recommended: Build The SIF Locally And Copy It To Sherlock

If Sherlock cannot build the image cleanly, build it on the local workstation
with Apptainer and copy the resulting SIF to Sherlock.

Local preflight:

```bash
apptainer --version
uname -m
command -v mksquashfs
command -v newuidmap || true
command -v newgidmap || true
df -h /home/yjhk/model-dev
ls -l /home/yjhk/model-dev/allatom-design/alphafold3/docker/jackhmmer_seq_limit.patch
```

Build locally with real root and explicit cache/temp locations:

```bash
mkdir -p /home/yjhk/model-dev/apptainer_tmp
mkdir -p /home/yjhk/model-dev/apptainer_cache
mkdir -p /home/yjhk/model-dev/logs

cd /home/yjhk/model-dev/allatom-design

sudo env \
  SIF=/home/yjhk/model-dev/phelix.sif \
  FORCE=1 \
  APPTAINER_BUILD_FLAGS= \
  APPTAINER_TMPDIR=/home/yjhk/model-dev/apptainer_tmp \
  APPTAINER_CACHEDIR=/home/yjhk/model-dev/apptainer_cache \
  bash /home/yjhk/model-dev/allatom-design/scripts/sherlock_scripts/jinho/setup/build_phelix_sif.sh \
  2>&1 | tee /home/yjhk/model-dev/logs/phelix_sif_build.log

sudo chown "$USER:$USER" /home/yjhk/model-dev/phelix.sif
```

Validate locally:

```bash
ls -lh /home/yjhk/model-dev/phelix.sif
apptainer exec /home/yjhk/model-dev/phelix.sif \
  bash -lc 'python3.12 --version && uv --version && gcc --version | head -1 && g++ --version | head -1 && jackhmmer -h | grep -- --seq_limit'
```

Copy to Sherlock:

```bash
ssh sherlock 'mkdir -p /scratch/users/zhkim216/containers'
rsync -avP /home/yjhk/model-dev/phelix.sif \
  sherlock:/scratch/users/zhkim216/containers/phelix.sif
```

Validate on Sherlock:

```bash
cd /home/users/zhkim216/code/phelix
git pull
apptainer exec --nv /scratch/users/zhkim216/containers/phelix.sif \
  bash -lc 'python3.12 --version && uv --version && jackhmmer -h | grep -- --seq_limit'
```

## Alternative: Build The SIF On Sherlock

The repo includes an Apptainer/Singularity definition that builds this image
from the NVIDIA CUDA 12.6 Ubuntu 24.04 Docker base and adds Python 3.12, uv,
zlib development headers, and patched HMMER:

```bash
cd /home/users/zhkim216/code/phelix
bash scripts/sherlock_scripts/jinho/setup/build_phelix_sif.sh
```

If Sherlock only has `/bin/singularity`, the build helper will use it
automatically. If Sherlock does not allow `--fakeroot`, retry with explicit
build flags:

```bash
APPTAINER_BUILD_FLAGS="" bash scripts/sherlock_scripts/jinho/setup/build_phelix_sif.sh
```

If an old or partial image already exists and you want to rebuild it:

```bash
FORCE=1 bash scripts/sherlock_scripts/jinho/setup/build_phelix_sif.sh
```

If the image is instead hosted in a Docker registry, pull it on Sherlock with:

```bash
mkdir -p /scratch/users/zhkim216/containers
apptainer pull /scratch/users/zhkim216/containers/phelix.sif docker://<your-phelix-image>
```

Check the built image:

```bash
apptainer exec /scratch/users/zhkim216/containers/phelix.sif \
  bash -lc 'python3.12 --version && uv --version && jackhmmer -h | grep -- --seq_limit'
```

Then update the repo:

```bash
cd /home/users/zhkim216/code/phelix
git pull
```

## Install

Enter the container:

```bash
cd /home/users/zhkim216/code/phelix
bash scripts/sherlock_scripts/jinho/setup/shell_in_container_phelix.sh
```

Inside the container, run:

```bash
bash scripts/sherlock_scripts/jinho/setup/install_phelix_sherlock.sh
```

The installer uses `requirements_split/sherlock` only for Torch/PyG pins. It
does not install `requirements_split/sherlock/*core*.txt`, because those files
belong to the old AlphaFold3 dependency set.

## Validate

Inside the container:

```bash
source /scratch/users/zhkim216/venv/phelix/bin/activate

python - <<'PY'
import jax, torch, rdkit, alphafold3, atomworks, allatom_design
print("jax", jax.__version__, jax.devices())
print("torch", torch.__version__, torch.cuda.is_available())
print("rdkit", rdkit.__version__)
print("alphafold3", alphafold3.__file__)
print("atomworks", atomworks.__file__)
PY

python alphafold3/run_alphafold_data_test.py \
  DataPipelineTest.test_template_chain_id_roundtrip \
  DataPipelineTest.test_ligand_template_conditioning_config \
  DataPipelineTest.test_ligand_template_conditioning_rejects_zero_templates
```

Expected `pip check` conflicts:

- `atomworks` declares `rdkit<2025.9`, while AlphaFold3-new uses `rdkit==2025.9.4`.
- `torch==2.7.0+cu126` declares `nvidia-cudnn-cu12==9.5.1.17`, while JAX 0.9.1 needs newer cuDNN. This setup pins `nvidia-cudnn-cu12==9.22.0.52`.

## Running Jobs

Wrap an existing sbatch script with:

```bash
bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_phelix.sh path/to/job.sbatch
```

The wrapper activates `/scratch/users/zhkim216/venv/phelix` inside
`/scratch/users/zhkim216/containers/phelix.sif` and runs the original job from
the mounted repo.
