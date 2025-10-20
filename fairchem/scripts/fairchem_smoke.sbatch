#!/usr/bin/env bash
#SBATCH -J fairchem_smoke
#SBATCH -p possu,bioe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/job_%A.out
#SBATCH --error=/scratch/users/zhkim216/job_output/job_%A.err

set -euo pipefail

# --- paths ---
SIF="/scratch/users/zhkim216/containers/allscaip.sif"
VENV="/scratch/users/zhkim216/venv/allscaip"
SING_HOME="$SCRATCH/.singhome_job"; mkdir -p "$SING_HOME"

export TORCH_HOME="$SCRATCH/cache/torch"
export HF_HOME="$SCRATCH/cache/huggingface"
export XDG_CACHE_HOME="$SCRATCH/cache/.cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/cache/inductor_cache"
export TRITON_CACHE_DIR="$SCRATCH/cache/triton_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH/cache/torch_extensions"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

echo "=== Node: $(hostname) | Job: ${SLURM_JOB_ID:-NA} ==="

# run
apptainer exec --nv --cleanenv \
  --home "$SING_HOME" \
  --bind "$SCRATCH:$SCRATCH" \
  "$SIF" /usr/bin/bash -lc "
  export TORCH_HOME='$TORCH_HOME'
  export HF_HOME='$HF_HOME'
  export XDG_CACHE_HOME='$XDG_CACHE_HOME'
  export TORCHINDUCTOR_CACHE_DIR='$TORCHINDUCTOR_CACHE_DIR'
  export TRITON_CACHE_DIR='$TRITON_CACHE_DIR'
  export TORCH_EXTENSIONS_DIR='$TORCH_EXTENSIONS_DIR'
  export PATH='$VENV/bin':\$PATH
  python - <<'PY'
import sys, torch, torch.backends.cudnn as cudnn
import e3nn, ase_db_backends
import fairchem.core as fc
print('Python :', sys.version.split()[0])
print('Torch  :', torch.__version__, '| CUDA:', torch.version.cuda, '| cuDNN:', cudnn.version())
x = torch.randn(4096, 4096, device='cuda'); y = x @ x.T
print('GPU OK :', float(y[0,0]))
print('e3nn   :', e3nn.__version__)
print('fairchem.core file:', fc.__file__)
PY
"
