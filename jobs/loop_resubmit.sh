#!/bin/bash
# Resubmit-loop for the 71 missing extrinsic eval cells.
# Reads /tmp/missing_jobs.txt (absolute PBS paths, one per line).
# Submits up to 45 at a time (ba39 gpuhopper cap = 50; keeps 5 headroom).
# Sleeps 5 min between passes; exits when the list is exhausted.
#
# Usage: nohup bash jobs/loop_resubmit.sh > logs/resubmit_missing_evals.log 2>&1 &

set -uo pipefail
LIST=/tmp/missing_jobs.txt
CAP=45
SLEEP_SEC=300

mkdir -p /g/data/ba39/dipankar/simt-tor-26/logs

# Track already-submitted (by job path)
declare -A SUBMITTED

TOTAL=$(wc -l < "$LIST")
echo "[resubmit] $(date) — pool: $TOTAL jobs"

while true; do
    remaining=0
    while IFS= read -r pbs; do
        [ -z "$pbs" ] && continue
        [ -n "${SUBMITTED[$pbs]:-}" ] && continue

        # Derive output-name from PBS to check landed
        out=$(grep -oE "(flores|wmt15|wmt22|iwslt17|iwslt15)_stream_v6b[a-zA-Z0-9]+_checkargmax_(low_medium|medium_high|low|medium|high)_[a-z]{2}-[a-z]{2}_n[0-9]+\.json" "$pbs" | head -1)
        if [ -n "$out" ] && [ -f "/g/data/ba39/dipankar/simt-tor-26/results/_archive/v6b_gemma_2b/extrinsic/$out" ]; then
            SUBMITTED[$pbs]=DONE
            continue
        fi

        # Check queue count on ba39
        q_count=$(qstat -u ds9561 2>/dev/null | grep -cE " ds9561 " || true)
        if [ "$q_count" -ge "$CAP" ]; then
            echo "[resubmit] $(date +%H:%M:%S) queue full ($q_count/$CAP) — sleeping ${SLEEP_SEC}s"
            break
        fi

        jid=$(qsub "$pbs" 2>&1)
        if [[ "$jid" =~ ^[0-9]+ ]]; then
            SUBMITTED[$pbs]="$jid"
            echo "[resubmit] $(date +%H:%M:%S) qsub $(basename $pbs) -> $jid"
        else
            echo "[resubmit] $(date +%H:%M:%S) qsub FAIL $(basename $pbs): $jid"
            # If queue-limit, break out and sleep; otherwise mark skip.
            if echo "$jid" | grep -q "exceed"; then
                break
            fi
            SUBMITTED[$pbs]=FAIL
        fi
        sleep 1
    done < "$LIST"

    # Count still-pending
    while IFS= read -r pbs; do
        [ -z "$pbs" ] && continue
        [ -z "${SUBMITTED[$pbs]:-}" ] && remaining=$((remaining+1))
    done < "$LIST"

    if [ "$remaining" -eq 0 ]; then
        echo "[resubmit] $(date) — all done."
        break
    fi
    echo "[resubmit] $(date +%H:%M:%S) $remaining still pending — sleeping ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
done
