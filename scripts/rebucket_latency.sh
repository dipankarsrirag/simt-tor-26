#!/usr/bin/env bash
# Post-annotation utility — rebucket latency labels in an SFT dataset
# using a different chunks/sent quantile split.
#
# Usage: bash scripts/rebucket_latency.sh --input {sft_dataset.json} --output {out.json}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/rebucket_latency.py" "$@"
