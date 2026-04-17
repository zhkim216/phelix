#!/bin/bash
# Submit the ligand-validity fetch + augment sbatch.
# Thin wrapper around fetch_and_augment_ligand_validity_v8.sbatch.
#
# Run after merge_and_augment_v8.sbatch has finished (and has produced
# metadata_augmented_lmpnn.parquet in the dataset dir).
#
# Usage:
#   bash submit_fetch_and_augment_ligand_validity_v8.sh            # submit
#   bash submit_fetch_and_augment_ligand_validity_v8.sh --dry-run  # print only
#
# Idempotent: re-running uses --skip-existing, so PDBs whose per-PDB
# cache JSON already exists are not re-fetched from RCSB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/fetch_and_augment_ligand_validity_v8.sbatch"
DATASET_DIR="${DATASET_DIR:-/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8}"
METADATA_IN="${DATASET_DIR}/metadata_augmented_lmpnn.parquet"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "ERROR: sbatch script not found at ${SBATCH_SCRIPT}" >&2
    exit 1
fi

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ ! -e "${METADATA_IN}" ]]; then
    cat <<EOF >&2
WARNING: expected input parquet not found: ${METADATA_IN}

This step needs metadata_augmented_lmpnn.parquet, which is produced by
  sbatch ${SCRIPT_DIR}/merge_and_augment_v8.sbatch

If that job hasn't run yet, launch it first. Continuing anyway so the
sbatch can still be submitted — the job itself will fail loudly if the
file is missing at run-time.
EOF
fi

CMD=(sbatch "${SBATCH_SCRIPT}")

echo "Command: ${CMD[*]}"
echo "  DATASET_DIR = ${DATASET_DIR}"
echo "  METADATA_IN = ${METADATA_IN}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run) not submitting."
    exit 0
fi

"${CMD[@]}"

cat <<'EOF'

Monitor with:
  squeue -u "$USER" | grep ligval_v8

After completion, verify:
  ls -la "${DATASET_DIR:-/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8}/metadata_augmented_lmpnn_ligval.parquet"
  du -sh "${DATASET_DIR:-/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8}/ligand_validity_cache_json"
EOF
