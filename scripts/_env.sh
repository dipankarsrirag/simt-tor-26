#!/usr/bin/env bash
# Shared env setup — sourced by every scripts/NN_*.sh wrapper.
# Portable across Gadi (H200) and laptops.
#
# Env vars you can override:
#   SIMT_REPO_ROOT   repo checkout (default: auto-detected from this file)
#   SIMT_VENV        path to Python venv (default: try Gadi, then repo-local)
#   SIMT_HF_CACHE    HuggingFace cache dir (default: Gadi path, else ~/.cache)
#   PYTHON           python binary (default: python3)

set -euo pipefail

# ─────────── repo root ───────────
export SIMT_REPO_ROOT="${SIMT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ─────────── PYTHONPATH ───────────
export PYTHONPATH="${SIMT_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ─────────── HF / model caches ───────────
# Priority: explicit SIMT_HF_CACHE → Gadi shared → $HOME/.cache/huggingface
if [ -n "${SIMT_HF_CACHE:-}" ]; then
  HF_CACHE="${SIMT_HF_CACHE}"
elif [ -d /g/data/po67/dipankar/cache ]; then
  HF_CACHE=/g/data/po67/dipankar/cache
else
  HF_CACHE="${HOME}/.cache/huggingface"
  mkdir -p "${HF_CACHE}/hub" "${HF_CACHE}/transformers"
fi
export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export TORCH_HOME="${HF_CACHE}/torch"
export XDG_CACHE_HOME="${HF_CACHE}"

# ─────────── venv activation ───────────
# If already inside a venv (VIRTUAL_ENV set), do nothing.
# Otherwise try SIMT_VENV → Gadi shared → repo-local .venv/ → skip.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  CANDIDATES=(
    "${SIMT_VENV:-}"
    "/scratch/po67/ds9561/.venv-fil"
    "${SIMT_REPO_ROOT}/.venv"
    "${SIMT_REPO_ROOT}/.venv-fil"
  )
  for venv in "${CANDIDATES[@]}"; do
    [ -z "$venv" ] && continue
    if [ -f "${venv}/bin/activate" ]; then
      # shellcheck disable=SC1090
      source "${venv}/bin/activate"
      break
    fi
  done
fi

# ─────────── python binary ───────────
export PYTHON="${PYTHON:-python3}"

# ─────────── report (opt-in via SIMT_ENV_VERBOSE=1) ───────────
if [ "${SIMT_ENV_VERBOSE:-0}" = "1" ]; then
  echo "[simt-env] SIMT_REPO_ROOT=${SIMT_REPO_ROOT}" >&2
  echo "[simt-env] HF_CACHE=${HF_CACHE}"             >&2
  echo "[simt-env] VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}" >&2
  echo "[simt-env] PYTHON=$(which ${PYTHON})"        >&2
fi
