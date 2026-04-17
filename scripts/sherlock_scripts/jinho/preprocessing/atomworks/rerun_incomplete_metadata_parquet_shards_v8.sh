#!/bin/bash
# Detect shards whose metadata_shard_{id:05d}.parquet is missing in
#   ${OUT_DIR}/shards, then (optionally) resubmit just those shards.
#
# build_metadata_parquet_shards_v8.sbatch is resume-safe: re-running with a
# smaller --array will union progress JSON + batch_*.parquet glob and skip
# batches already on disk, so failed shards finish where they left off.
#
# Behavior:
#   1. Run detect_incomplete_shards.py --verbose --write-report.
#      JSON report is saved to ${OUT_DIR}/detection_reports/detection_<timestamp>.json.
#   2. Print the most recent .err file for each incomplete shard so you
#      can inspect stderr before deciding on resource overrides.
#   3. If --submit is passed, resubmit only the incomplete shards via
#        sbatch --array=<incomplete> [extra sbatch flags] <build_sbatch>
#      Extra --<flag> <value> pairs are forwarded to sbatch verbatim, so
#      --mem, --time, --cpus-per-task, --partition, etc. all work.
#   4. If --merge is passed *and* all shards are complete, submit
#      merge_and_augment_v8.sbatch.
#
# Usage:
#   # Detection only (safe default):
#   bash rerun_incomplete_metadata_parquet_shards_v8.sh
#
#   # Detect + resubmit failed shards with default resources:
#   bash rerun_incomplete_metadata_parquet_shards_v8.sh --submit
#
#   # Detect + resubmit with more RAM and longer wall-time:
#   bash rerun_incomplete_metadata_parquet_shards_v8.sh --submit --mem 64GB --time 2-00:00:00
#
#   # After all shards complete, chain merge_and_augment:
#   bash rerun_incomplete_metadata_parquet_shards_v8.sh --merge
#
# Environment overrides:
#   OUT_DIR       default /scratch/users/zhkim216/datasets/atomworks_pdb_full_v8
#   NUM_SHARDS    default 100
#   JOB_OUT_DIR   default /scratch/users/zhkim216/job_output/data_preprocessing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: scripts/sherlock_scripts/jinho/preprocessing/atomworks/ -> up 5 levels.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# -- Defaults (override via env or CLI) --
OUT_DIR="${OUT_DIR:-/scratch/users/zhkim216/datasets/atomworks_pdb_full_v8}"
NUM_SHARDS="${NUM_SHARDS:-100}"
BUILD_SBATCH="${SCRIPT_DIR}/build_metadata_parquet_shards_v8.sbatch"
MERGE_SBATCH="${SCRIPT_DIR}/merge_and_augment_v8.sbatch"
JOB_OUT_DIR="${JOB_OUT_DIR:-/scratch/users/zhkim216/job_output/data_preprocessing}"

SUBMIT=0
DO_MERGE=0
EXTRA_SBATCH=()

# --submit / --merge are consumed; --out_dir / --num-shards are honored;
# everything else starting with -- is forwarded to sbatch.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) SUBMIT=1; shift ;;
        --merge)  DO_MERGE=1; shift ;;
        --out_dir) OUT_DIR="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,39p' "$0"
            exit 0
            ;;
        --*=*)
            # e.g. --mem=64GB -> forward as-is
            EXTRA_SBATCH+=("$1")
            shift
            ;;
        --*)
            # e.g. --mem 64GB -> forward flag + value
            if [[ $# -ge 2 && "$2" != --* ]]; then
                EXTRA_SBATCH+=("$1" "$2")
                shift 2
            else
                EXTRA_SBATCH+=("$1")
                shift
            fi
            ;;
        *)
            echo "Unknown positional arg: $1" >&2
            exit 2
            ;;
    esac
done

REPORTS_DIR="${OUT_DIR}/detection_reports"
mkdir -p "${REPORTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_JSON="${REPORTS_DIR}/detection_${TIMESTAMP}.json"

echo "=== Detect incomplete shards ==="
echo "  out_dir    = ${OUT_DIR}"
echo "  num_shards = ${NUM_SHARDS}"
echo "  report     = ${REPORT_JSON}"
echo

cd "${REPO_ROOT}"
python3 -m allatom_design.data.preprocessing.atomworks.detect_incomplete_shards \
    --out_dir "${OUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --job-output-dir "${JOB_OUT_DIR}" \
    --verbose \
    --write-report "${REPORT_JSON}"

SLURM_ARRAY="$(python3 -c "import json; d=json.load(open('${REPORT_JSON}')); print(d.get('slurm_array',''))")"
INCOMPLETE_COUNT="$(python3 -c "import json; d=json.load(open('${REPORT_JSON}')); print(d.get('incomplete_count',0))")"

echo
if [[ "${INCOMPLETE_COUNT}" == "0" ]]; then
    echo "All shards complete; nothing to resubmit."
    if [[ "${DO_MERGE}" -eq 1 ]]; then
        echo
        echo "=== Submitting merge_and_augment_v8.sbatch ==="
        sbatch "${MERGE_SBATCH}"
    else
        cat <<EOF

Next step: launch the merge pipeline either by:
  sbatch ${MERGE_SBATCH}
or rerun this script with --merge.
EOF
    fi
    exit 0
fi

echo "Incomplete shards (${INCOMPLETE_COUNT}): slurm array = ${SLURM_ARRAY}"
echo
echo "Inspect job output for each failed array id at:"
echo "  ${JOB_OUT_DIR}/build_metadata_parquet_shards_v8_<JOB_ID>_<ARRAY>.{out,err}"
echo
IFS=',' read -ra ARRAY_RANGES <<< "${SLURM_ARRAY}"
echo "Most recent .err files (sorted by mtime):"
for range in "${ARRAY_RANGES[@]}"; do
    if [[ "${range}" == *-* ]]; then
        start="${range%-*}"; end="${range#*-}"
        seq_ids=$(seq "${start}" "${end}")
    else
        seq_ids="${range}"
    fi
    for id in ${seq_ids}; do
        latest="$(ls -t "${JOB_OUT_DIR}"/build_metadata_parquet_shards_v8_*_"${id}".err 2>/dev/null | head -n1 || true)"
        if [[ -n "${latest}" ]]; then
            echo "  shard ${id}: ${latest}"
        else
            echo "  shard ${id}: (no .err file found)"
        fi
    done
done
echo

if [[ "${SUBMIT}" -ne 1 ]]; then
    cat <<EOF

Detection-only mode. When ready to resubmit, rerun:
  bash $(basename "$0") --submit [sbatch overrides, e.g. --mem 64GB --time 2-00:00:00]

Optionally chain merge_and_augment (only acts once all shards complete):
  bash $(basename "$0") --submit --merge

Report saved to: ${REPORT_JSON}
EOF
    exit 0
fi

CMD=(sbatch --array="${SLURM_ARRAY}")
if [[ ${#EXTRA_SBATCH[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_SBATCH[@]}")
fi
CMD+=("${BUILD_SBATCH}")

echo "=== Resubmitting failed shards ==="
echo "Command: ${CMD[*]}"
"${CMD[@]}"

if [[ "${DO_MERGE}" -eq 1 ]]; then
    echo
    cat <<EOF
The build resubmission above will take wall-time to finish. Once it's done
(and this script shows 'All shards complete'), submit:
  sbatch ${MERGE_SBATCH}
Or rerun:
  bash $(basename "$0") --merge
EOF
fi

cat <<EOF

Report saved to: ${REPORT_JSON}
Next: monitor with 'squeue -u "\$USER" | grep metadata_atomworks_pdb_full_v8',
      then re-run this script (without --submit) to confirm all shards complete.
EOF
