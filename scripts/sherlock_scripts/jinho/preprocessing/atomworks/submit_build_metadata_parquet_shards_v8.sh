#!/bin/bash
# Submit the initial v8 build sbatch (--array=0-99).
# Thin wrapper around build_metadata_parquet_shards_v8.sbatch.
#
# Skip this if the initial build has already been submitted — the
# rerun_incomplete_metadata_parquet_shards_v8.sh script will discover
# any shards that did not finish regardless of how the first submission
# was made.
#
# Usage:
#   bash submit_build_metadata_parquet_shards_v8.sh            # submit
#   bash submit_build_metadata_parquet_shards_v8.sh --dry-run  # print only
#
# Monitor with:
#   squeue -u "$USER" | grep metadata_atomworks_pdb_full_v8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/build_metadata_parquet_shards_v8.sbatch"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "ERROR: sbatch script not found at ${SBATCH_SCRIPT}" >&2
    exit 1
fi

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

CMD=(sbatch "${SBATCH_SCRIPT}")

echo "Command: ${CMD[*]}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run) not submitting."
    exit 0
fi

"${CMD[@]}"

cat <<'EOF'

Submission complete. Monitor with:
  squeue -u "$USER" | grep metadata_atomworks_pdb_full_v8

Job output:
  /scratch/users/zhkim216/job_output/data_preprocessing/build_metadata_parquet_shards_v8_<JOB_ID>_<ARRAY>.{out,err}

After the array finishes, run:
  bash rerun_incomplete_metadata_parquet_shards_v8.sh
EOF
