#!/usr/bin/env bash
# Sanity-check that a new backbone is drop-in compatible with our pipeline
# (chat template + EOR/EOW tokens + latency instruction format).
#
# Usage: bash scripts/probe_east_8b_compat.sh --model_dir /path/to/model

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/probe_east_8b_compat.py" "$@"
