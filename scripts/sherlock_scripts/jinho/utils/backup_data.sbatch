#!/bin/bash
#SBATCH --job-name=backup
#SBATCH -p possu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --time=6-0:00:00
#SBATCH --mem=32GB
#SBATCH --output=/scratch/users/zhkim216/job_output/backup/job_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/backup/job_%A_%a.err

TARGET_DIR="/oak/stanford/groups/possu/jinho/260212_scratch_backup"
mkdir -p "${TARGET_DIR}"

# Resolve compressor: prefer pigz (parallel), fall back to gzip.
if ! command -v pigz >/dev/null 2>&1; then
    module load system pigz >/dev/null 2>&1 || module load pigz >/dev/null 2>&1 || true
fi
if command -v pigz >/dev/null 2>&1; then
    COMPRESS_CMD=(pigz -p "${SLURM_CPUS_PER_TASK}")
    echo "Using pigz ($(command -v pigz)) with ${SLURM_CPUS_PER_TASK} threads."
else
    COMPRESS_CMD=(gzip)
    echo "WARNING: pigz not available; falling back to single-threaded gzip."
fi

# Define source directories (index matches SLURM_ARRAY_TASK_ID)
DIRS=(
    "/scratch/users/zhkim216/out_dir/ligand_seq_des"       # 0
    "/scratch/users/zhkim216/out_dir/eval_ligand_seq_des"  # 1
    "/scratch/users/zhkim216/af3_databases"                # 2
    "/scratch/users/zhkim216/containers"                   # 3
    "/scratch/users/zhkim216/mmCIF2BioLiP"                 # 4
    "/scratch/users/zhkim216/Plinder"                      # 5
    "/scratch/users/zhkim216/PoseX"                        # 6
    "/scratch/users/zhkim216/datasets"                     # 7
    "/scratch/users/zhkim216/model_params"                 # 8
    "/scratch/users/zhkim216/Potts_GFP"                    # 9
    "/scratch/users/zhkim216/out_dir"                      # 10 (top-level, if needed)
)

SRC_DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "${SRC_DIR}" ]; then
    echo "ERROR: No directory mapped for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: Directory does not exist: ${SRC_DIR}"
    exit 1
fi

# Derive archive name from directory basename
# For nested dirs like out_dir/ligand_seq_des, join with underscore
ARCHIVE_NAME=$(echo "${SRC_DIR}" | sed 's|/scratch/users/zhkim216/||' | tr '/' '_')
TAR_FILE="/scratch/users/zhkim216/${ARCHIVE_NAME}.tar.gz"

echo "=== Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Source:  ${SRC_DIR}"
echo "Archive: ${TAR_FILE}"
echo "Target:  ${TARGET_DIR}/"
echo "Start:   $(date)"

# Step 1: tar + compress
echo "[Step 1] Creating tar.gz with ${COMPRESS_CMD[0]}..."
tar -cf - -C "$(dirname "${SRC_DIR}")" "$(basename "${SRC_DIR}")" \
    | "${COMPRESS_CMD[@]}" > "${TAR_FILE}"
PIPE_STATUS=("${PIPESTATUS[@]}")
if [ "${PIPE_STATUS[0]}" -ne 0 ] || [ "${PIPE_STATUS[1]}" -ne 0 ]; then
    echo "ERROR: tar+${COMPRESS_CMD[0]} failed (tar=${PIPE_STATUS[0]} compress=${PIPE_STATUS[1]})"
    rm -f "${TAR_FILE}"
    exit 1
fi

echo "Archive size: $(du -h "${TAR_FILE}" | cut -f1)"

# Step 2: Move to oak
echo "[Step 2] Moving archive to oak..."
mv "${TAR_FILE}" "${TARGET_DIR}/"

if [ $? -ne 0 ]; then
    echo "ERROR: mv to oak failed"
    exit 1
fi

# Step 3: Verify the file exists on oak, then delete scratch tar (already moved)
if [ -f "${TARGET_DIR}/$(basename "${TAR_FILE}")" ]; then
    echo "[Step 3] Archive verified on oak."
else
    echo "ERROR: Archive not found on oak after mv!"
    exit 1
fi

echo "Done:    $(date)"