#!/bin/bash
#PBS -N ${JOBNAME}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
${NGPUS_LINE}
#PBS -l walltime=${WALLTIME}
#PBS -l storage=${STORAGE}
#PBS -l wd
#PBS -j oe
#PBS -o ${LOG_FILE}

# Auto-resubmit wrapper for long SFT / annotation jobs.
# Ported from ../arabic-dial-mt/pbs/templates/job.pbs.tpl.
#
# Contract: the wrapped script writes exactly one of DONE / NEEDS_RESUME
# to $OUTPUT_DIR before exit. FAILED is written here on any other exit.
# See HOUSEKEEPING.md §6.6.

set -eo pipefail
umask 022

# Cache + venv activation lives in a project env script.
source "/g/data/ba39/dipankar/simt-tor-26/pbs/env.sh"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

if [[ -f "${OUTPUT_DIR}/DONE" ]]; then
    echo "[$(date -Iseconds)] DONE marker present; nothing to do."
    exit 0
fi
if [[ -f "${OUTPUT_DIR}/FAILED" ]]; then
    echo "[$(date -Iseconds)] FAILED marker present; refusing to run. Clear it after diagnosis."
    exit 1
fi

RESUBMIT_COUNT_FILE="${OUTPUT_DIR}/.resubmit_count"
RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)
MAX_RESUBMITS=10

echo "[$(date -Iseconds)] job=$PBS_JOBID queue=${QUEUE} resubmit=$RESUBMIT_COUNT/$MAX_RESUBMITS"
echo "[$(date -Iseconds)] output_dir=${OUTPUT_DIR}"
echo "[$(date -Iseconds)] script=${SCRIPT_ABS} config=${CONFIG_PATH}"

_finalize_on_unexpected_exit() {
    if [[ ! -f "${OUTPUT_DIR}/DONE" && ! -f "${OUTPUT_DIR}/NEEDS_RESUME" && ! -f "${OUTPUT_DIR}/FAILED" ]]; then
        touch "${OUTPUT_DIR}/FAILED"
        echo "[$(date -Iseconds)] Unexpected exit; wrote FAILED."
    fi
}
trap _finalize_on_unexpected_exit EXIT

python -u "${SCRIPT_ABS}" --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[$(date -Iseconds)] Script exited $EXIT_CODE without any marker; marking FAILED."
    touch "${OUTPUT_DIR}/FAILED"
    trap - EXIT
    exit $EXIT_CODE
fi

if [[ -f "${OUTPUT_DIR}/DONE" ]]; then
    echo "[$(date -Iseconds)] DONE marker written; run complete."
    trap - EXIT
    exit 0
fi

if [[ -f "${OUTPUT_DIR}/NEEDS_RESUME" ]]; then
    if (( RESUBMIT_COUNT >= MAX_RESUBMITS )); then
        echo "[$(date -Iseconds)] Resubmit cap ($MAX_RESUBMITS) reached; marking FAILED."
        touch "${OUTPUT_DIR}/FAILED"
        trap - EXIT
        exit 1
    fi
    echo $((RESUBMIT_COUNT + 1)) > "$RESUBMIT_COUNT_FILE"
    rm -f "${OUTPUT_DIR}/NEEDS_RESUME"
    NEW_JOBID=$(qsub "$0")
    echo "[$(date -Iseconds)] Resubmitted as $NEW_JOBID"
    trap - EXIT
    exit 0
fi

echo "[$(date -Iseconds)] Script exited 0 with no marker; treating as anomaly."
touch "${OUTPUT_DIR}/FAILED"
trap - EXIT
exit 1
