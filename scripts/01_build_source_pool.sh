#!/usr/bin/env bash
# Stage 1 — build the source pool for a given experiment tag.
# Reads: configs/{tag}.yaml
# Writes: results/sft_dataset/{tag}/source_pool.json
#         results/sft_dataset/{tag}/per_direction/{pair}.json
#         results/sft_dataset/{tag}/build_manifest.json
#
# Usage: bash scripts/01_build_source_pool.sh --config configs/{tag}.yaml
#        (any additional args are passed through to the Python script)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/01_build_source_pool.py" "$@"
