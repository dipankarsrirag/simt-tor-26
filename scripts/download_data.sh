#!/bin/bash
# download_data.sh — pull all Phase 0 datasets onto po67 gdata.
# Runs on copyq (has internet); assumes .venv-fil is active.
# See HOUSEKEEPING.md §3.

set -euo pipefail

DATA_ROOT=/g/data/po67/dipankar/data/simt-tor-26
mkdir -p "$DATA_ROOT"

echo "=== [1/3] SiMT-De-En-660K (biaofu-xmu, GPT-4-tagged parallel) ==="
if [[ -d "$DATA_ROOT/SiMT-De-En-660K" && -n "$(ls -A "$DATA_ROOT/SiMT-De-En-660K" 2>/dev/null)" ]]; then
    echo "already present at $DATA_ROOT/SiMT-De-En-660K — skipping"
else
    hf download biaofu-xmu/SiMT-De-En-660K \
        --repo-type dataset \
        --local-dir "$DATA_ROOT/SiMT-De-En-660K"
    du -sh "$DATA_ROOT/SiMT-De-En-660K"
fi

echo
echo "=== [2/3] WMT15 De-En newstest2015 (via sacrebleu) ==="
mkdir -p "$DATA_ROOT/wmt15-de-en"
if [[ -f "$DATA_ROOT/wmt15-de-en/newstest2015.de" && -f "$DATA_ROOT/wmt15-de-en/newstest2015.en" ]]; then
    echo "already present — skipping"
else
    sacrebleu -t wmt15 -l de-en --echo src > "$DATA_ROOT/wmt15-de-en/newstest2015.de"
    sacrebleu -t wmt15 -l de-en --echo ref > "$DATA_ROOT/wmt15-de-en/newstest2015.en"
    wc -l "$DATA_ROOT/wmt15-de-en/"*.de "$DATA_ROOT/wmt15-de-en/"*.en
fi

echo
echo "=== [3/3] RWTH De-En gold alignments (EAST Appendix E.4) ==="
# EAST Appendix E.4 uses the RWTH manually-aligned De-En corpus for the
# intrinsic annotation-quality measure. Current canonical source needs
# confirmation from EAST's paper — the release URL is not stable across
# years. Leaving as TODO for Phase 0.
#
# When confirmed, uncomment and adjust:
#   curl -L -o "$DATA_ROOT/rwth-de-en.tar.gz" "<URL from EAST §E.4>"
#   tar -xzf "$DATA_ROOT/rwth-de-en.tar.gz" -C "$DATA_ROOT/"
echo "TODO — confirm RWTH source from EAST paper before adding here."
echo "See HOUSEKEEPING §3 and TIMELINE.md Phase 0."

echo
echo "=== Done ==="
ls -la "$DATA_ROOT"
