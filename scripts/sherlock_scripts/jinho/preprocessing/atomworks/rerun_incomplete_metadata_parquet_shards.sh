#!/bin/bash
# Count missing metadata_shard_{id:05d}.parquet files under ${DATASET_DIR}/shards.
# Prints a comma-separated array spec (e.g. "3,5,12-14") ready to paste into
# `#SBATCH --array=` of build_metadata_parquet_shards_vN.sbatch for a rerun.
#
# Usage: bash rerun_incomplete_metadata_parquet_shards.sh <dataset_dir> <num_shards>

set -euo pipefail

DATASET_DIR="${1:?Usage: $0 <dataset_dir> <num_shards>}"
NUM_SHARDS="${2:?Usage: $0 <dataset_dir> <num_shards>}"

SHARDS_DIR="${DATASET_DIR}/shards"
if [[ ! -d "${SHARDS_DIR}" ]]; then
    echo "ERROR: shards dir not found: ${SHARDS_DIR}" >&2
    exit 2
fi

missing=()
for ((i=0; i<NUM_SHARDS; i++)); do
    f="${SHARDS_DIR}/metadata_shard_$(printf '%05d' "$i").parquet"
    [[ -f "$f" ]] || missing+=("$i")
done

echo "incomplete: ${#missing[@]} / ${NUM_SHARDS}"

if (( ${#missing[@]} == 0 )); then
    exit 0
fi

# Collapse consecutive ids into ranges (e.g. 0-2,5,7-9).
python3 - "${missing[@]}" <<'PY'
import sys
ids = sorted(int(x) for x in sys.argv[1:])
out, i = [], 0
while i < len(ids):
    j = i
    while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
        j += 1
    out.append(str(ids[i]) if j == i else f"{ids[i]}-{ids[j]}")
    i = j + 1
print(",".join(out))
PY
