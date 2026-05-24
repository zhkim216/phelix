#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

REPO_ROOT="${HOME}/code/allatom-design"
SCRIPT_DIR="${REPO_ROOT}/scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/denovoval_sweep/scaffold_generation_remaining"
WRAP_SCRIPT="${REPO_ROOT}/scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container.sh"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT not found: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${WRAP_SCRIPT}" ]]; then
  echo "ERROR: wrapper script not found: ${WRAP_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${SCRIPT_DIR}" ]]; then
  echo "ERROR: script dir not found: ${SCRIPT_DIR}" >&2
  exit 1
fi

mapfile -t SBATCH_FILES < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f -name '*.sbatch' | sort)
if [[ ${#SBATCH_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no sbatch files found in ${SCRIPT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

LOG_FILE="${SCRIPT_DIR}/submitted_jobs_$(date +%Y%m%d_%H%M%S).log"
echo "Submitting ${#SBATCH_FILES[@]} jobs" | tee -a "${LOG_FILE}"

for sbatch_abs in "${SBATCH_FILES[@]}"; do
  sbatch_rel="${sbatch_abs#${REPO_ROOT}/}"
  cmd=(bash "${WRAP_SCRIPT}" "${sbatch_rel}")
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] ${cmd[*]}" | tee -a "${LOG_FILE}"
  else
    echo "[SUBMIT] ${cmd[*]}" | tee -a "${LOG_FILE}"
    "${cmd[@]}" | tee -a "${LOG_FILE}"
  fi
done

echo "Done. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
