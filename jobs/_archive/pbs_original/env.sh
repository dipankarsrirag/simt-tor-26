#!/bin/bash
# Sourced by every PBS job. Sets HF cache, offline mode, PYTHONPATH, venv.
# Idempotent: safe to source multiple times.
#
# Ported from ../arabic-dial-mt/pbs/env.sh with path swaps for simt-tor-26.
# Cache lives on the shared po67 gdata cache — persistent across scratch
# purges and shared with sibling projects.

REPO_ROOT="${SIMT_REPO_ROOT:-/g/data/ba39/dipankar/simt-tor-26}"
VENV="${SIMT_VENV:-/scratch/po67/ds9561/.venv-fil}"
HF_CACHE="${SIMT_HF_CACHE:-/g/data/po67/dipankar/cache}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TORCH_HOME="${HF_CACHE}/torch"
export TRITON_CACHE_DIR="${HF_CACHE}/triton"
export XDG_CACHE_HOME="${HF_CACHE}"
export PIP_CACHE_DIR="${HF_CACHE}/pip"
export TMPDIR="${HF_CACHE}/tmp"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Compute nodes have no internet; only copyq should hit HF.
_queue="${PBS_QUEUE:-}"
if [[ "${_queue}" != "copyq" && "${_queue}" != "copyq-exec" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

# Bind CPU threads to the allocated cores.
if [[ -n "${PBS_NCPUS:-}" ]]; then
    export OMP_NUM_THREADS="${PBS_NCPUS}"
    export MKL_NUM_THREADS="${PBS_NCPUS}"
fi

module load python3/3.10.4
source "${VENV}/bin/activate"
