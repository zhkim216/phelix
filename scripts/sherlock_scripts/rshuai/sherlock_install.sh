# # Pull apptainer image
# mkdir -p $GROUP_HOME/containers
# apptainer pull $GROUP_HOME/containers/pytorch_24.12.sif docker://nvcr.io/nvidia/pytorch:24.12-py3

# Set paths
export IMG="$GROUP_HOME/containers/pytorch_24.12.sif"
export UV_CACHE_DIR="$SCRATCH/uv/cache"
export REPO_DIR="/home/users/zhkim216/code/allatom-design"
export ENV_DIR="$SCRATCH/venv"
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
'

# # Install colabdesign.
# uv pip install colabdesign@git+https://github.com/sokrypton/ColabDesign.git@d024c4e846fea83c090afcbe89a313eeee8ec01e
# uv pip install optax==0.1.7
# uv pip install flax==0.12.0
# uv pip install "jax[cuda12]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# uv pip install numpy==1.26.3

# # Install Protpardelle.
# rm -rf protpardelle-1c  # ensure protpardelle-1c is not already in the repo
# git clone https://github.com/ProteinDesignLab/protpardelle-1c.git
# cd protpardelle-1c
# git checkout 7ebffddcd3b25b986ed36fccebce9676f50ce676
# cd ..
# uv pip install -e protpardelle-1c --no-deps