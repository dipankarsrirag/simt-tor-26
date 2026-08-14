#!/bin/bash
# download_data.sh — pull all Phase 0 datasets onto po67 gdata.
# Runs on copyq (has internet); assumes .venv-fil is active.
# See HOUSEKEEPING.md §3 and EXPERIMENTS.md for how each dataset is used.
#
# Idempotent — safe to re-run; already-present datasets are skipped.

set -euo pipefail

DATA_ROOT=/g/data/po67/dipankar/data/simt-tor-26
mkdir -p "$DATA_ROOT"

# --- SFT data ----------------------------------------------------------------
# EAST recipe (Fu et al. 2025 §3.1, §3.2):
#   Stage I : full-weight SFT on SiMT-De-En-660K  (WMT15 De->En training,
#             GPT-4 chunk annotations at low/medium/high latency).
#   Stage II: LoRA on SiMT-Multi-90K (8 directions) + Off-Multi-120K (OMT).
# We scope to Stage I as the primary run (see LOG.md decision).
# Multi-90K is fetched anyway — cheap and enables the multilingual stretch.

echo "=== [1/6] SiMT-De-En-660K (Stage-I SFT, De->En, GPT-4-tagged) ==="
if [[ -d "$DATA_ROOT/SiMT-De-En-660K" && -n "$(ls -A "$DATA_ROOT/SiMT-De-En-660K" 2>/dev/null)" ]]; then
    echo "already present — skipping"
else
    hf download biaofu-xmu/SiMT-De-En-660K \
        --repo-type dataset \
        --local-dir "$DATA_ROOT/SiMT-De-En-660K"
    du -sh "$DATA_ROOT/SiMT-De-En-660K"
fi

echo
echo "=== [2/6] SiMT-Multi-90K (Stage-II SFT, 8 directions, GPT-4-tagged) ==="
if [[ -d "$DATA_ROOT/SiMT-Multi-90K" && -n "$(ls -A "$DATA_ROOT/SiMT-Multi-90K" 2>/dev/null)" ]]; then
    echo "already present — skipping"
else
    hf download biaofu-xmu/SiMT-Multi-90K \
        --repo-type dataset \
        --local-dir "$DATA_ROOT/SiMT-Multi-90K"
    du -sh "$DATA_ROOT/SiMT-Multi-90K"
fi

# --- Test sets ----------------------------------------------------------------
# WMT15 De->En newstest2015 — EAST Fig. 3, our primary sentence-level SiMT test.
# WMT22 X<->En (8 directions) — EAST Fig. 4, stretch for multilingual runs.
# WMT22 also supplies docid, which lets us do document-level SiMT (EAST §4.3).

echo
echo "=== [3/6] WMT15 De->En newstest2015 (primary SiMT test) ==="
mkdir -p "$DATA_ROOT/wmt15-de-en"
if [[ -f "$DATA_ROOT/wmt15-de-en/newstest2015.de" && -f "$DATA_ROOT/wmt15-de-en/newstest2015.en" ]]; then
    echo "already present — skipping"
else
    sacrebleu -t wmt15 -l de-en --echo src > "$DATA_ROOT/wmt15-de-en/newstest2015.de"
    sacrebleu -t wmt15 -l de-en --echo ref > "$DATA_ROOT/wmt15-de-en/newstest2015.en"
    wc -l "$DATA_ROOT/wmt15-de-en/"*.de "$DATA_ROOT/wmt15-de-en/"*.en
fi

echo
echo "=== [4/6] WMT22 X<->En 8 directions (multilingual stretch test) ==="
mkdir -p "$DATA_ROOT/wmt22"
for pair in de-en en-de zh-en en-zh ru-en en-ru cs-en en-cs; do
    OUT_DIR="$DATA_ROOT/wmt22/$pair"
    mkdir -p "$OUT_DIR"
    SRC_L=${pair%-*}
    TGT_L=${pair#*-}
    if [[ -f "$OUT_DIR/newstest2022.$SRC_L" && -f "$OUT_DIR/newstest2022.$TGT_L" && -f "$OUT_DIR/newstest2022.docid" ]]; then
        echo "  $pair: already present"
    else
        sacrebleu -t wmt22 -l "$pair" --echo src   > "$OUT_DIR/newstest2022.$SRC_L"
        sacrebleu -t wmt22 -l "$pair" --echo ref   > "$OUT_DIR/newstest2022.$TGT_L"
        sacrebleu -t wmt22 -l "$pair" --echo docid > "$OUT_DIR/newstest2022.docid"
        echo "  $pair: $(wc -l < "$OUT_DIR/newstest2022.$SRC_L") sentences"
    fi
done

# --- Intrinsic annotation-quality eval ----------------------------------------
# EAST Appendix E.4 uses the RWTH manually-aligned De->En corpus to score
# annotation quality directly (proportion of gold-aligned source tokens read
# before each target token). Independent of both annotators — cleanest evidence.
# The canonical release URL is not stable across years. Check EAST paper §E.4
# for the current source, drop it in below, and re-run.

echo
echo "=== [5/6] RWTH De->En gold alignments (intrinsic eval, EAST App. E.4) ==="
echo "TODO — confirm URL from EAST paper before adding. See LOG.md."

# --- Off-Multi-120K (only needed if we do Stage II) --------------------------
# Reconstructed from WMT17-21 test data following ALMA (Xu et al. 2024a),
# 8 directions (De/Zh/Ru/Cs <-> En). Not published as a single HF dataset —
# has to be assembled from WMT17-21 via sacrebleu. Skipped by default; enable
# only if Gate 3 in TIMELINE.md passes and we attempt the multilingual stretch.

echo
echo "=== [6/6] Off-Multi-120K (Stage-II OMT, WMT17-21, ALMA-style) ==="
echo "SKIPPED — assemble only if attempting the multilingual stretch (post-Gate 3)."
echo "See scripts/build_off_multi.py (TODO)."

echo
echo "=== Done ==="
ls -la "$DATA_ROOT"
du -sh "$DATA_ROOT"/*
