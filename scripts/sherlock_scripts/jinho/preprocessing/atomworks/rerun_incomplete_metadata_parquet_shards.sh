#!/bin/bash
# Detect shards whose metadata_shard_{id:05d}.parquet is missing under
# ${OUT_DIR}/shards, optionally resubmit just those shards.
#
# Version-agnostic: takes the path to a build_metadata_parquet_shards_*.sbatch
# as its first positional argument and derives everything else from it:
#   NUM_SHARDS        ← #SBATCH --array=<spec>        (max array id + 1)
#   OUT_DIR           ← out_dir=... on the python CLI
#   JOB_OUT_DIR       ← dirname of #SBATCH --output=...
#   ERR_FILE_TEMPLATE ← basename of #SBATCH --error=..., with %A→* and %a→{shard_id}
#
# build_metadata_parquet_shards_*.sbatch is resume-safe: rerunning with a
# smaller --array union-merges progress JSON + batch_*.parquet on disk and
# skips batches already completed, so failed shards finish where they left off.
#
# Behavior:
#   1. Parse the sbatch to derive NUM_SHARDS / OUT_DIR / JOB_OUT_DIR /
#      ERR_FILE_TEMPLATE, then run detect_incomplete_shards.py --verbose
#      --write-report. JSON report lands in
#      ${OUT_DIR}/detection_reports/detection_<timestamp>.json.
#   2. Print the most recent .err file for each incomplete shard (using the
#      template derived from the sbatch) so stderr can be inspected before
#      overriding resources.
#   3. If --submit is passed, resubmit only the incomplete shards via
#        sbatch --array=<incomplete> [extra sbatch flags] <build_sbatch>
#      Any --<flag> <value> or --<flag>=<value> not consumed below is
#      forwarded verbatim, so --mem / --time / --cpus-per-task / --partition
#      all work.
#
# Usage:
#   # Detection only (safe default):
#   bash rerun_incomplete_metadata_parquet_shards.sh <build_sbatch>
#
#   # Detect + resubmit failed shards with default resources:
#   bash rerun_incomplete_metadata_parquet_shards.sh <build_sbatch> --submit
#
#   # Detect + resubmit with more RAM and longer wall-time:
#   bash rerun_incomplete_metadata_parquet_shards.sh <build_sbatch> --submit \
#       --mem 64GB --time 2-00:00:00
#
# Merge step (now out of scope for this script):
#   Once all shards are complete, merge directly via
#     python3 allatom_design/data/preprocessing/atomworks/merge_parquet_shards.py \
#         --out_dir <OUT_DIR>
#
# Flag overrides (rare — override values auto-parsed from the sbatch):
#   --out_dir PATH         override auto-detected OUT_DIR
#   --num-shards N         override auto-detected NUM_SHARDS
#   --job-output-dir PATH  override auto-detected JOB_OUT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: scripts/sherlock_scripts/jinho/preprocessing/atomworks/ -> up 5 levels.
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

BUILD_SBATCH=""
SUBMIT=0
EXTRA_SBATCH=()
# Optional overrides — stay empty unless user sets them explicitly.
OUT_DIR_OVERRIDE=""
NUM_SHARDS_OVERRIDE=""
JOB_OUT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)          SUBMIT=1; shift ;;
        --out_dir)         OUT_DIR_OVERRIDE="$2"; shift 2 ;;
        --num-shards)      NUM_SHARDS_OVERRIDE="$2"; shift 2 ;;
        --job-output-dir)  JOB_OUT_DIR_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,46p' "$0"
            exit 0
            ;;
        --*=*)
            # e.g. --mem=64GB -> forward as-is to sbatch
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
            if [[ -z "${BUILD_SBATCH}" ]]; then
                BUILD_SBATCH="$1"
                shift
            else
                echo "Unexpected positional arg: $1 (build sbatch already set to ${BUILD_SBATCH})" >&2
                exit 2
            fi
            ;;
    esac
done

if [[ -z "${BUILD_SBATCH}" ]]; then
    echo "Usage: bash $(basename "$0") <build_sbatch> [--submit] [sbatch overrides]" >&2
    echo "Run with -h for details." >&2
    exit 2
fi
if [[ ! -f "${BUILD_SBATCH}" ]]; then
    echo "Build sbatch not found: ${BUILD_SBATCH}" >&2
    exit 2
fi

# -- Parse the sbatch --
# Python does the parsing because array specs like '0-49%10' or '1,3,5-7' need
# real logic, and shlex.quote lets us round-trip paths safely through eval.
SBATCH_VARS="$(python3 - "${BUILD_SBATCH}" <<'PYEOF'
import pathlib
import re
import shlex
import sys

text = pathlib.Path(sys.argv[1]).read_text()

def find_sbatch_directive(key: str) -> str | None:
    m = re.search(rf'^\s*#SBATCH\s+--{key}=(\S+)', text, re.MULTILINE)
    return m.group(1) if m else None

def max_array_id(spec: str) -> int:
    # Strip concurrency suffix (e.g. '0-49%10' -> '0-49').
    spec = spec.split('%', 1)[0]
    max_id = -1
    for part in spec.split(','):
        endpoints = part.split('-')
        for ep in endpoints:
            try:
                max_id = max(max_id, int(ep))
            except ValueError:
                pass
    if max_id < 0:
        raise SystemExit(f"Could not parse --array spec: {spec!r}")
    return max_id

array_spec = find_sbatch_directive('array')
if not array_spec:
    raise SystemExit("No '#SBATCH --array=...' directive found")
num_shards = max_array_id(array_spec) + 1

m = re.search(r'(?:^|\s)out_dir=(\S+)', text)
if not m:
    raise SystemExit("No 'out_dir=...' found on the python command line")
out_dir = m.group(1)

output_spec = find_sbatch_directive('output')
if not output_spec:
    raise SystemExit("No '#SBATCH --output=...' directive found")
job_out_dir = str(pathlib.Path(output_spec).parent)

error_spec = find_sbatch_directive('error')
if not error_spec:
    raise SystemExit("No '#SBATCH --error=...' directive found")
err_basename = pathlib.Path(error_spec).name
# SLURM expands %A → job id, %a → array task id. detect_incomplete_shards
# uses '*' for the former (any job id) and '{shard_id}' for the latter.
err_template = err_basename.replace('%A', '*').replace('%a', '{shard_id}')

print(f"PARSED_NUM_SHARDS={shlex.quote(str(num_shards))}")
print(f"PARSED_OUT_DIR={shlex.quote(out_dir)}")
print(f"PARSED_JOB_OUT_DIR={shlex.quote(job_out_dir)}")
print(f"PARSED_ERR_FILE_TEMPLATE={shlex.quote(err_template)}")
PYEOF
)"

eval "${SBATCH_VARS}"

OUT_DIR="${OUT_DIR_OVERRIDE:-${PARSED_OUT_DIR}}"
NUM_SHARDS="${NUM_SHARDS_OVERRIDE:-${PARSED_NUM_SHARDS}}"
JOB_OUT_DIR="${JOB_OUT_DIR_OVERRIDE:-${PARSED_JOB_OUT_DIR}}"
ERR_FILE_TEMPLATE="${PARSED_ERR_FILE_TEMPLATE}"

REPORTS_DIR="${OUT_DIR}/detection_reports"
mkdir -p "${REPORTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_JSON="${REPORTS_DIR}/detection_${TIMESTAMP}.json"

echo "=== Detect incomplete shards ==="
echo "  build_sbatch      = ${BUILD_SBATCH}"
echo "  out_dir           = ${OUT_DIR}"
echo "  num_shards        = ${NUM_SHARDS}"
echo "  job_output_dir    = ${JOB_OUT_DIR}"
echo "  err_file_template = ${ERR_FILE_TEMPLATE}"
echo "  report            = ${REPORT_JSON}"
echo

cd "${REPO_ROOT}"
python3 -m allatom_design.data.preprocessing.atomworks.detect_incomplete_shards \
    --out_dir "${OUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --job-output-dir "${JOB_OUT_DIR}" \
    --err-file-template "${ERR_FILE_TEMPLATE}" \
    --verbose \
    --write-report "${REPORT_JSON}"

SLURM_ARRAY="$(python3 -c "import json; d=json.load(open('${REPORT_JSON}')); print(d.get('slurm_array',''))")"
INCOMPLETE_COUNT="$(python3 -c "import json; d=json.load(open('${REPORT_JSON}')); print(d.get('incomplete_count',0))")"

echo
if [[ "${INCOMPLETE_COUNT}" == "0" ]]; then
    cat <<EOF
All shards complete; nothing to resubmit.

Next step — merge shards:
  python3 allatom_design/data/preprocessing/atomworks/merge_parquet_shards.py \\
      --out_dir ${OUT_DIR}
EOF
    exit 0
fi

echo "Incomplete shards (${INCOMPLETE_COUNT}): slurm array = ${SLURM_ARRAY}"
echo
echo "Inspect job output for each failed array id at:"
echo "  ${JOB_OUT_DIR}/${ERR_FILE_TEMPLATE//\{shard_id\}/<ARRAY>} (.out/.err)"
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
        pattern="${ERR_FILE_TEMPLATE//\{shard_id\}/${id}}"
        latest="$(ls -t "${JOB_OUT_DIR}"/${pattern} 2>/dev/null | head -n1 || true)"
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
  bash $(basename "$0") ${BUILD_SBATCH} --submit [sbatch overrides, e.g. --mem 64GB --time 2-00:00:00]

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

cat <<EOF

Report saved to: ${REPORT_JSON}
Next: monitor with 'squeue -u "\$USER"', then re-run this script (without --submit)
      to confirm all shards complete. Once complete, merge with:
  python3 allatom_design/data/preprocessing/atomworks/merge_parquet_shards.py \\
      --out_dir ${OUT_DIR}
EOF
