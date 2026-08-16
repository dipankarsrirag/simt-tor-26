#!/bin/bash
# Runs the Gate-1 reordering-bin analysis on both landed sweeps.
# Preconditions:
#   - results/gate1/gpt4_pearson_full.json (from phase1_precompute_gpt4_pearson.py)
#   - results/phase1_tau_sweep_ot_n200/matrices.jsonl (from OT sweep)
#   - results/phase1_tau_sweep_js_n200/matrices.jsonl (from JS sweep)
set -eo pipefail

cd /g/data/ba39/dipankar/simt-tor-26
source /scratch/po67/ds9561/.venv-fil/bin/activate

echo "=============================================="
echo "OT n=200 — winning config (base + raw + OT)"
echo "=============================================="
python -u scripts/phase1_reordering_bin.py \
  --matrices results/phase1_tau_sweep_ot_n200/matrices.jsonl \
  --gpt4_pearson_full results/gate1/gpt4_pearson_full.json \
  --tau_grid 0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00 \
  --output results/gate1/reordering_bin_ot_n200.json

echo ""
echo "=============================================="
echo "JS n=200 — cheap-criterion ablation"
echo "=============================================="
python -u scripts/phase1_reordering_bin.py \
  --matrices results/phase1_tau_sweep_js_n200/matrices.jsonl \
  --gpt4_pearson_full results/gate1/gpt4_pearson_full.json \
  --tau_grid 0.02,0.05,0.08,0.10,0.15,0.20,0.30 \
  --output results/gate1/reordering_bin_js_n200.json

echo ""
echo "Reports written:"
echo "  results/gate1/reordering_bin_ot_n200.json"
echo "  results/gate1/reordering_bin_js_n200.json"
