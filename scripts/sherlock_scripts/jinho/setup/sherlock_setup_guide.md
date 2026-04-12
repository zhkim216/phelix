# Sherlock Environment Setup Guide

Commands to run after installing the lcaliby venv.

## Prerequisites

- Access to Sherlock cluster
- lcaliby venv installed at `/scratch/users/zhkim216/venv/lcaliby`
- lcaliby.sif container at `/scratch/users/zhkim216/containers/lcaliby.sif`
- Schrodinger installed at `/scratch/users/zhkim216/software/schrodinger2025-3`

---

## 1. Clone & checkout

```bash
cd /home/users/zhkim216/code
git clone <repo-url> allatom-design
cd allatom-design
git checkout jinho/AAA
```

---

## 2. Copy system libraries to oak (one-time)

The container (Ubuntu 24.04) is missing system libraries that Schrodinger PrepWizard/Glide need:
`libglib-2.0.so.0`, `libgthread-2.0.so.0`, `libpcre.so.1`.
Copy them from the Sherlock bare metal host to oak.

**Run outside the container (bare metal):**

```bash
mkdir -p /oak/stanford/groups/possu/jinho/libs

cp /usr/lib64/libglib-2.0.so.0 /oak/stanford/groups/possu/jinho/libs/
cp /usr/lib64/libgthread-2.0.so.0 /oak/stanford/groups/possu/jinho/libs/
cp /usr/lib64/libpcre.so.1 /oak/stanford/groups/possu/jinho/libs/
```

**Verify:**

```bash
ls -la /oak/stanford/groups/possu/jinho/libs/
# Should show libglib-2.0.so.0, libgthread-2.0.so.0, libpcre.so.1
```

> Stored on oak, so survives scratch purges. Only needs to be done once.

---

## 3. Generate machine-id (one-time)

Sherlock compute nodes are diskless, so `/etc/machine-id` is empty.
Schrodinger SLM (License Manager) requires this file. Generate a fake one
and store it under `/home` (persistent across purges). The container scripts
automatically bind-mount it as `--bind $MACHINE_ID_FILE:/etc/machine-id:ro`.

```bash
mkdir -p /home/users/zhkim216/.schrodinger
python3 -c "import uuid; print(uuid.uuid4().hex)" > /home/users/zhkim216/.schrodinger/machine-id
cat /home/users/zhkim216/.schrodinger/machine-id
# Should print a 32-char hex string (e.g. a1b2c3d4e5f6...)
```

> Only needs to be done once.

---

## 4. Verify Schrodinger license file

Schrodinger SLM expects a FlexLM-format `lmgrd.lic`.

```bash
cat /scratch/users/zhkim216/software/schrodinger2025-3/licenses/lmgrd.lic
```

Should contain:
```
SERVER srcc-license-srcf.stanford.edu ANY 53001
USE_SERVER
```

If different, restore it:
```bash
cat > /scratch/users/zhkim216/software/schrodinger2025-3/licenses/lmgrd.lic << 'EOF'
SERVER srcc-license-srcf.stanford.edu ANY 53001
USE_SERVER
EOF
```

---

## 5. Create cache directories

```bash
mkdir -p /scratch/users/zhkim216/cache/{torch,huggingface,pip_cache,.cache,.pycache}
mkdir -p /scratch/users/zhkim216/cache/{inductor_cache,triton_cache,torch_extensions}
mkdir -p /scratch/users/zhkim216/cache/jax_compilation_cache
mkdir -p /scratch/users/zhkim216/uv/{cache,python}
```

---

## 6. Verify environment scripts

All environment variables are set automatically by `env_setup.sh`.
No manual exports needed.

```bash
cd /home/users/zhkim216/code/allatom-design

# Load env_setup.sh (run on bare metal)
source scripts/sherlock_scripts/jinho/setup/env_setup.sh

# Check key variables
echo "SCHRODINGER: $SCHRODINGER"
echo "SCHRODINGER_LD_LIBS: $SCHRODINGER_LD_LIBS"
echo "OAK_LIBS: $OAK_LIBS"
echo "MACHINE_ID_FILE: $MACHINE_ID_FILE"
echo "SIF: $SIF"
echo "VENV: $VENV"
```

Expected output:
```
SCHRODINGER: /scratch/users/zhkim216/software/schrodinger2025-3
SCHRODINGER_LD_LIBS: /scratch/users/zhkim216/software/schrodinger2025-3/internal/lib:...:/oak/stanford/groups/possu/jinho/libs
OAK_LIBS: /oak/stanford/groups/possu/jinho/libs
MACHINE_ID_FILE: /home/users/zhkim216/.schrodinger/machine-id
SIF: /scratch/users/zhkim216/containers/lcaliby.sif
VENV: /scratch/users/zhkim216/venv/lcaliby
```

---

## 7. Interactive shell test

### 7-1. Allocate a node

```bash
srun -p possu,owners --cpus-per-task=8 --mem=16G --time=02:00:00 --pty bash
```

### 7-2. Enter the container (auto-configured)

```bash
cd /home/users/zhkim216/code/allatom-design
bash scripts/sherlock_scripts/jinho/setup/shell_in_container.sh
```

This script automatically handles:
- CUDA bind mount
- `LD_LIBRARY_PATH` (CUDA + Schrodinger + oak libraries)
- `SCHRODINGER`, `SCHROD_LICENSE_FILE`
- `/etc/machine-id` bind mount for SLM
- `PYTHONPATH`, cache directories, etc.

### 7-3. Verify inside the container

```bash
# Activate venv
source /scratch/users/zhkim216/venv/lcaliby/bin/activate

# Check Python
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# Check PoseBusters patch
python3 -c "
from allatom_design.eval.eval_utils.eval_posebusters import __patched
print(f'PB monkey-patch applied: {__patched}')
"

# Check Schrodinger
echo "SCHRODINGER=$SCHRODINGER"
cat /etc/machine-id
$SCHRODINGER/utilities/prepwizard -h 2>&1 | head -5
```

---

## 8. Glide/PB pipeline verification

Inside the container with venv activated:

```bash
cd /home/users/zhkim216/code/allatom-design
export PYTHONPATH=$(pwd):$PYTHONPATH

# PB only (no Glide) -- should complete without segfault
python3 -m allatom_design.eval.glide.run_ligand_eval_batch \
    --config-path /home/users/zhkim216/code/allatom-design/allatom_design/configs/eval/glide \
    --config-name run_ligand_eval_batch \
    af3_eval_dir=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_af3/step_82500_epoch_52 \
    output_dir=/scratch/users/zhkim216/out_dir/test_setup_verification \
    sample_dir=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs/designable \
    debug=true \
    num_debug_samples=2 \
    num_workers=1 \
    glide.enabled=false \
    hydra.run.dir=/scratch/users/zhkim216/hydra_outputs

# Full pipeline (PB + Glide)
python3 -m allatom_design.eval.glide.run_ligand_eval_batch \
    --config-path /home/users/zhkim216/code/allatom-design/allatom_design/configs/eval/glide \
    --config-name run_ligand_eval_batch \
    af3_eval_dir=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp35_cfg2_denovoval_designable_af3/step_82500_epoch_52 \
    output_dir=/scratch/users/zhkim216/out_dir/test_setup_verification_full \
    sample_dir=/scratch/users/zhkim216/datasets/val_cifs/denovo_val_cifs/designable \
    schrodinger.schrodinger_path=/scratch/users/zhkim216/software/schrodinger2025-3 \
    debug=true \
    num_debug_samples=2 \
    num_workers=1 \
    hydra.run.dir=/scratch/users/zhkim216/hydra_outputs
```

---

## 9. sbatch job submission test

Run outside the container (bare metal):

```bash
cd /home/users/zhkim216/code/allatom-design

# wrap_sbatch_in_container.sh auto-injects the environment and submits via sbatch
bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container.sh \
    scripts/sherlock_scripts/jinho/eval_seq_des_training/ligand_eval/ligand_eval_exp35_cfg2_denovoval.sbatch
```

---

## File structure reference

```
scripts/sherlock_scripts/jinho/setup/
  env_setup.sh                 # Master env config (sourced by all scripts)
  schrodinger_env.sh           # Schrodinger paths + LD_LIBRARY_PATH + license + machine-id
  shell_in_container.sh        # Interactive container shell
  wrap_sbatch_in_container.sh  # Wraps sbatch scripts to run inside container
```

**Env chain**: `env_setup.sh` -> `schrodinger_env.sh` (sourced) -> all container scripts use these automatically.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: libglib-2.0.so.0` | Oak libs not copied or bind mount missing | Redo Step 2 |
| `ImportError: libpcre.so.1` | libpcre missing from oak libs | `cp /usr/lib64/libpcre.so.1 /oak/.../libs/` |
| PoseBusters segfault | Monkey-patch not applied | `eval_posebusters.py` auto-patches on import -- check `__patched` |
| `SCHRODINGER not set` | `schrodinger_env.sh` not sourced | Check that `env_setup.sh` sources it |
| PrepWizard license error | License server not configured or machine-id missing | Check `$SCHROD_LICENSE_FILE` and `cat /etc/machine-id` inside container |
| Empty `/etc/machine-id` in container | machine-id bind mount missing | Redo Step 3, verify `$MACHINE_ID_FILE` is set |
| `licenseFileType is not a member` | lmgrd.lic has wrong format (e.g. JSON) | Redo Step 4 to restore FlexLM format |
| `module: command not found` (inside container) | Container has no module system -- this is normal | `env_setup.sh` runs on bare metal, vars passed via `--env` |

---

## Background: why this setup is needed

1. **PoseBusters segfault**: rdkit pip wheel (2025.9.6, 2026.3.1) bug. `GetSubstructMatches` with aromatic SMARTS causes C++ Conformer memory corruption -> segfault. The monkey-patch in `eval_posebusters.py` fixes this automatically.

2. **Missing Schrodinger libraries**: The Ubuntu 24.04 container lacks `libglib-2.0`, `libgthread-2.0`, `libpcre`. Provided via `LD_LIBRARY_PATH` from Schrodinger bundled libs + host system libs copied to oak.

3. **Empty machine-id on diskless nodes**: Sherlock compute nodes are network-booted, so `/etc/machine-id` is empty. Schrodinger SLM requires a valid machine-id. A generated one is bind-mounted into the container.

4. **Automation**: `schrodinger_env.sh` -> `env_setup.sh` -> all container scripts. No manual exports needed.
