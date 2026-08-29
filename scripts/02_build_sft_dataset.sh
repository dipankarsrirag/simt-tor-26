#!/usr/bin/env bash
# Stage 3 — build the SFT dataset from annotator matrices.
# Reads: configs/{tag}.yaml, results/annotate/{tag}/matrices.jsonl
# Writes: results/sft_dataset/{tag}/sft_dataset.json
#
# Usage: bash scripts/02_build_sft_dataset.sh --config configs/{tag}.yaml

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/02_build_sft_dataset.py" "$@"
