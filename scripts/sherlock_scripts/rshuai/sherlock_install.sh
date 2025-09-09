# # Pull apptainer image
# mkdir -p $GROUP_HOME/containers
# apptainer pull $GROUP_HOME/containers/pytorch_25.08.sif docker://nvcr.io/nvidia/pytorch:25.08-py3

# Set paths
export IMG="$GROUP_HOME/containers/pytorch_25.08.sif"
export UV_CACHE_DIR="$SCRATCH/cache"
export REPO_DIR="/home/users/rshuai/code/allatom-design"
export ENV_DIR="$SCRATCH/envs"
mkdir -p "$ENV_DIR"

# Install dependencies within apptainer
apptainer exec --nv \
  --bind $HOME:$HOME \
  --bind $UV_CACHE_DIR:$UV_CACHE_DIR \
  --bind $GROUP_HOME:$GROUP_HOME \
  --bind $REPO_DIR:$REPO_DIR \
  "$IMG" bash -lc '
set -euo pipefail

# create & activate venv in $SCRATCH
ENV_DIR="'"$ENV_DIR"'"
mkdir -p "$ENV_DIR"
cd "$ENV_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv allatom_design -p python3.12
source allatom_design/bin/activate

cd "'"$REPO_DIR"'"
uv pip sync uv_indexes.txt uv.lock --index-strategy=unsafe-best-match --cache-dir '"$UV_CACHE_DIR"'
uv pip install -e .

# maybe risky but if needed, to install colabdesign
uv pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install colabdesign@git+https://github.com/sokrypton/ColabDesign.git@d024c4e846fea83c090afcbe89a313eeee8ec01e
'
