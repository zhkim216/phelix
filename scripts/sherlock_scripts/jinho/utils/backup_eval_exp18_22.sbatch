#!/bin/bash
#SBATCH --job-name=backup_eval_exp18_22
#SBATCH -p possu
#SBATCH --array=0-29%5
#SBATCH --cpus-per-task=16
#SBATCH --time=1-0:00:00
#SBATCH --mem=32GB
#SBATCH --output=/scratch/users/zhkim216/job_output/backup/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/backup/job_%A_%a.err

set -u

SRC_PARENT="/scratch/users/zhkim216/out_dir/eval_ligand_seq_des"
TARGET_DIR="/oak/stanford/groups/possu/jinho/260417_scratch_backup"
SCRATCH_TMP="/scratch/users/zhkim216"

# Resolve compressor: prefer pigz (parallel), fall back to gzip.
if ! command -v pigz >/dev/null 2>&1; then
    module load system pigz >/dev/null 2>&1 || module load pigz >/dev/null 2>&1 || true
fi
if command -v pigz >/dev/null 2>&1; then
    COMPRESS_CMD=(pigz -p "${SLURM_CPUS_PER_TASK}")
    VERIFY_CMD=(pigz -t)
    echo "Using pigz ($(command -v pigz)) with ${SLURM_CPUS_PER_TASK} threads."
else
    COMPRESS_CMD=(gzip)
    VERIFY_CMD=(gzip -t)
    echo "WARNING: pigz not available; falling back to single-threaded gzip."
fi

mkdir -p "${TARGET_DIR}"
mkdir -p "$(dirname /scratch/users/zhkim216/job_output/backup/.)"

# Discover source directories matching eval_exp{18..22}_*
shopt -s nullglob
DIRS=()
for prefix in eval_exp18_ eval_exp19_ eval_exp20_ eval_exp21_ eval_exp22_; do
    for d in "${SRC_PARENT}/${prefix}"*; do
        [ -d "$d" ] && DIRS+=("$d")
    done
done

N=${#DIRS[@]}
echo "Discovered ${N} directories matching eval_exp18..22 under ${SRC_PARENT}"

if [ "${N}" -eq 0 ]; then
    echo "ERROR: no matching directories found."
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N}" ]; then
    echo "Task ID ${SLURM_ARRAY_TASK_ID} is beyond directory count ${N}. Nothing to do."
    exit 0
fi

SRC_DIR="${DIRS[${SLURM_ARRAY_TASK_ID}]}"
BASENAME="$(basename "${SRC_DIR}")"
TAR_FILE="${SCRATCH_TMP}/${BASENAME}.tar.gz"
TARGET_FILE="${TARGET_DIR}/${BASENAME}.tar.gz"

echo "=== Task ${SLURM_ARRAY_TASK_ID} / ${N} ==="
echo "Source : ${SRC_DIR}"
echo "Archive: ${TAR_FILE}"
echo "Target : ${TARGET_FILE}"
echo "Start  : $(date)"

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: Source directory disappeared: ${SRC_DIR}"
    exit 1
fi

if [ -f "${TARGET_FILE}" ]; then
    echo "NOTE: ${TARGET_FILE} already exists; will overwrite after successful build."
fi

# Step 1: tar + compress to scratch
echo "[Step 1] Creating tar.gz with ${COMPRESS_CMD[0]}..."
rm -f "${TAR_FILE}"
tar -cf - -C "$(dirname "${SRC_DIR}")" "${BASENAME}" \
    | "${COMPRESS_CMD[@]}" > "${TAR_FILE}"
PIPE_STATUS=("${PIPESTATUS[@]}")
if [ "${PIPE_STATUS[0]}" -ne 0 ] || [ "${PIPE_STATUS[1]}" -ne 0 ]; then
    echo "ERROR: tar+${COMPRESS_CMD[0]} failed (tar=${PIPE_STATUS[0]} compress=${PIPE_STATUS[1]})"
    rm -f "${TAR_FILE}"
    exit 1
fi
echo "Archive size: $(du -h "${TAR_FILE}" | cut -f1)"
LOCAL_SIZE=$(stat -c %s "${TAR_FILE}")

# Step 2: Local integrity check
echo "[Step 2] Verifying archive integrity on scratch..."
if ! "${VERIFY_CMD[@]}" "${TAR_FILE}"; then
    echo "ERROR: ${VERIFY_CMD[0]} integrity check failed on scratch archive."
    rm -f "${TAR_FILE}"
    exit 1
fi
if ! tar -tzf "${TAR_FILE}" > /dev/null; then
    echo "ERROR: tar listing failed on scratch archive."
    rm -f "${TAR_FILE}"
    exit 1
fi

# Step 3: Move to oak
echo "[Step 3] Moving archive to oak..."
mv -f "${TAR_FILE}" "${TARGET_FILE}"
if [ $? -ne 0 ]; then
    echo "ERROR: mv to oak failed"
    exit 1
fi

# Step 4: Verify on oak
echo "[Step 4] Verifying archive on oak..."
if [ ! -f "${TARGET_FILE}" ]; then
    echo "ERROR: Archive not found on oak after mv."
    exit 1
fi
OAK_SIZE=$(stat -c %s "${TARGET_FILE}")
if [ "${OAK_SIZE}" -ne "${LOCAL_SIZE}" ]; then
    echo "ERROR: size mismatch (local=${LOCAL_SIZE}, oak=${OAK_SIZE})."
    exit 1
fi
if ! "${VERIFY_CMD[@]}" "${TARGET_FILE}"; then
    echo "ERROR: ${VERIFY_CMD[0]} integrity check failed on oak archive."
    exit 1
fi
if ! tar -tzf "${TARGET_FILE}" > /dev/null; then
    echo "ERROR: tar listing failed on oak archive."
    exit 1
fi
echo "Archive verified on oak (size=${OAK_SIZE} bytes)."

# Step 5: Delete source
echo "[Step 5] Deleting source directory: ${SRC_DIR}"
rm -rf "${SRC_DIR}"
if [ $? -ne 0 ] || [ -d "${SRC_DIR}" ]; then
    echo "ERROR: failed to delete source directory."
    exit 1
fi

echo "Done   : $(date)"
