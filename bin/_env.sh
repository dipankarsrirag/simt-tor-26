#!/usr/bin/env bash
# Shared env setup — sourced by every bin/* wrapper.
# Portable: no site-specific hardcoded paths. All defaults are $HOME-based.
#
# Env vars you can override (in ~/.bashrc, ~/.zshrc, or ad-hoc):
#   SIMT_REPO_ROOT      repo checkout          (default: auto-detected from this file)
#   SIMT_VENV           Python venv            (default: ./.venv → ./.venv-fil → system Python)
#   SIMT_HF_CACHE       HuggingFace cache      (default: $HOME/.cache/huggingface)
#   SIMT_MODEL_BASE     model-weights root     (default: $HOME/.cache/simt-models)
#   SIMT_DATA_ROOT      parallel-corpora root  (default: $SIMT_REPO_ROOT/data)
#   SIMT_CORPUS_ROOT    training-data root     (default: $SIMT_DATA_ROOT)
#   SIMT_TESTSETS_ROOT  eval test-set root     (default: $SIMT_DATA_ROOT)
#   PYTHON              python binary          (default: python3)
#
# Site-specific paths (e.g. Gadi shared caches) go in your shell rc file:
#   export SIMT_MODEL_BASE=/g/data/po67/dipankar/models
#   export SIMT_HF_CACHE=/g/data/po67/dipankar/cache

set -euo pipefail

# ─────────── repo root ───────────
export SIMT_REPO_ROOT="${SIMT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ─────────── filesystem conventions ───────────
export SIMT_DATA_ROOT="${SIMT_DATA_ROOT:-${SIMT_REPO_ROOT}/data}"
export SIMT_CORPUS_ROOT="${SIMT_CORPUS_ROOT:-${SIMT_DATA_ROOT}}"
export SIMT_TESTSETS_ROOT="${SIMT_TESTSETS_ROOT:-${SIMT_DATA_ROOT}}"
export SIMT_MODEL_BASE="${SIMT_MODEL_BASE:-${HOME}/.cache/simt-models}"
mkdir -p "${SIMT_MODEL_BASE}"

# ─────────── PYTHONPATH ───────────
export PYTHONPATH="${SIMT_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ─────────── HF / torch caches ───────────
export SIMT_HF_CACHE="${SIMT_HF_CACHE:-${HOME}/.cache/huggingface}"
mkdir -p "${SIMT_HF_CACHE}/hub" "${SIMT_HF_CACHE}/transformers"
export HF_HOME="${SIMT_HF_CACHE}"
export HF_HUB_CACHE="${SIMT_HF_CACHE}/hub"
export HUGGINGFACE_HUB_CACHE="${SIMT_HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${SIMT_HF_CACHE}/transformers"
export TORCH_HOME="${SIMT_HF_CACHE}/torch"
export XDG_CACHE_HOME="${SIMT_HF_CACHE}"

# ─────────── venv activation ───────────
# If already inside a venv (VIRTUAL_ENV set), do nothing.
# Otherwise try SIMT_VENV → ./.venv → ./.venv-fil → skip (use system Python).
if [ -z "${VIRTUAL_ENV:-}" ]; then
  CANDIDATES=(
    "${SIMT_VENV:-}"
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
  echo "[simt-env] SIMT_REPO_ROOT=${SIMT_REPO_ROOT}"     >&2
  echo "[simt-env] SIMT_DATA_ROOT=${SIMT_DATA_ROOT}"     >&2
  echo "[simt-env] SIMT_MODEL_BASE=${SIMT_MODEL_BASE}"   >&2
  echo "[simt-env] SIMT_HF_CACHE=${SIMT_HF_CACHE}"       >&2
  echo "[simt-env] VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}"   >&2
  echo "[simt-env] PYTHON=$(which ${PYTHON})"            >&2
fi
