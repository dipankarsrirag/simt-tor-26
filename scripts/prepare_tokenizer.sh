#!/usr/bin/env bash
# One-time — extend a backbone's tokenizer with EOR/EOW special tokens.
# Reads: HF backbone
# Writes: results/train/{tag}/tokenizer/  (or path from --output)
#
# Usage: bash scripts/prepare_tokenizer.sh --backbone google/gemma-4-E2B-it --output results/train/{tag}/tokenizer/

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${DIR}/_env.sh"
exec "${PYTHON}" -u "${DIR}/prepare_tokenizer.py" "$@"
