#!/bin/bash
# create-venv.sh — install simt-tor-26's Python-package extras.
#
# simt-tor-26 shares its Python environment with the first-impressions-last
# and simul-mt projects. The base env lives at /scratch/po67/ds9561/.venv-fil
# and provides torch, transformers, tokenizers, vllm, huggingface_hub,
# sentencepiece, numpy, safetensors, flashinfer — the inference stack.
#
# This script tops up that shared env with the packages simt-tor-26 needs
# on top of fil's base. Idempotent — safe to re-run; uv skips satisfied deps.
#
# Usage:
#     bash create-venv.sh
#
# If .venv-fil is missing (post scratch purge), rebuild the fil venv first
# from ../first-impressions-last/, then re-run this script.
# See HOUSEKEEPING.md §4.

set -euo pipefail

VENV=/scratch/po67/ds9561/.venv-fil
UV=/g/data/po67/dipankar/uv/bin/uv

if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: base venv missing at $VENV" >&2
    echo "Rebuild the fil venv first (see ../first-impressions-last/)," >&2
    echo "then re-run this script." >&2
    exit 1
fi

if [ ! -x "$UV" ]; then
    echo "ERROR: uv not found at $UV" >&2
    exit 1
fi

# simt-tor-26 extras on top of the fil base env.
# Add new deps to this list as they come up and re-run.
#
# Annotator (METHOD.md §§1–4):
#   pot          — Python Optimal Transport (Sinkhorn) for the OT criterion
#
# SFT (EXPERIMENTS.md primary result):
#   trl          — SFTTrainer for the annotated read/write data
#   accelerate   — distributed launch helper
#   peft         — LoRA if we go there; harmless otherwise
#   datasets     — HF datasets (SiMT-De-En-660K loader)
#
# Eval:
#   sacrebleu    — BLEU with pinned tokenisers
#
# Baselines (docs/followup-experiments.md §Fig 2/3):
#   awesome-align — word-level source↔target alignments for Wang 2024
#                   conv-simt baseline (annotator for scripts/07_conv.py).
#                   Needs `bert-base-multilingual-cased` on disk at
#                   MODEL_BASE/mBERT (fetch via /g/data/po67/dipankar/
#                   models/get_model.py).
"$UV" pip install --python "$VENV/bin/python" \
    pot \
    trl \
    accelerate \
    peft \
    datasets \
    sacrebleu \
    awesome-align

echo
echo ">>> Done. Activate with:  source $VENV/bin/activate"
echo ">>> Freeze:               $UV pip freeze --python $VENV/bin/python > .venv-freeze.txt"
