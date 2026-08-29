#!/usr/bin/env bash
# Stage 6 — plot BLEU-vs-AL curves for the landed eval matrix.
# Reads: results/eval/*/*.json (or results/_archive/*/extrinsic/*.json)
# Writes: figures/{tag}/*.png
#
# Usage: bash scripts/03_plot_bleu_al.sh [--config configs/plots.yaml]

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/03_plot_bleu_al.py" "$@"
