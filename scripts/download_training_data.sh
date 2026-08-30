#!/usr/bin/env bash
# Download the CURATED training corpus (europarl-v10 + news-commentary-v16 +
# TED2020) for all 8 language pairs used in the follow-up experiments.
#
# Output layout (matches scripts/01_build_source_pool.py's expectations):
#
#   ${SIMT_CORPUS_ROOT}/  (defaults to ${SIMT_DATA_ROOT})
#   ├── parallel_clean/
#   │   ├── de-en/
#   │   │   ├── europarl.de       europarl.en
#   │   │   ├── news-commentary.de news-commentary.en
#   │   │   └── ted2020.de        ted2020.en
#   │   ├── ru-en/
#   │   │   ├── news-commentary.en news-commentary.ru
#   │   │   └── ted2020.en        ted2020.ru
#   │   └── ar-en/
#   │       ├── ted2020.ar        ted2020.en
#   └── raw/
#       └── ted2020-en-vi/
#           ├── TED2020.en-vi.vi
#           └── TED2020.en-vi.en
#
# Sources (all human-translated, verified 0% overlap with FLORES devtest):
#   Europarl-v10          European Parliament proceedings (Koehn 2005)
#   news-commentary-v16   WMT17-21 shared task
#   TED2020               OPUS moses tarballs, TED Open Translation Project
#                         (Reimers & Gurevych 2020)
#
# Total download size: ~4.5GB compressed, ~12GB extracted.
# Skips any file/directory already present. Idempotent.
#
# Usage:
#   bash scripts/download_training_data.sh
#   # or with custom output root:
#   SIMT_CORPUS_ROOT=/path/to/data bash scripts/download_training_data.sh

set -euo pipefail

ROOT="${SIMT_CORPUS_ROOT:-${SIMT_DATA_ROOT:-$(pwd)/data}}"
PARALLEL="${ROOT}/parallel_clean"
RAW="${ROOT}/raw"
mkdir -p "$PARALLEL" "$RAW"

echo "═════════════════════════════════════════════════════════════════"
echo "  Downloading curated training corpus"
echo "  → ${ROOT}"
echo "═════════════════════════════════════════════════════════════════"

# ─── helper: fetch + extract an OPUS moses zip if not already there ─────
fetch_opus() {
  local url="$1"     # e.g. https://opus.nlpl.eu/TED2020/de&en/v1/moses/de-en.txt.zip
  local out_dir="$2" # where to extract
  local expect="$3"  # sample filename inside the zip; skip if present

  mkdir -p "$out_dir"
  if [ -f "${out_dir}/${expect}" ]; then
    echo "  [OPUS] $expect already present — skipping"
    return
  fi
  local tmp="/tmp/opus_$$.zip"
  echo "  [OPUS] $url"
  curl -fL "$url" -o "$tmp"
  unzip -q -o "$tmp" -d "$out_dir"
  rm -f "$tmp"
  ls "${out_dir}/"*.{en,de,ru,ar,vi} 2>/dev/null | head -3
}

# ─── helper: fetch statmt tsv or tar ─────────────────────────────────────
fetch_statmt() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "  [statmt] $(basename "$out") already present — skipping"
    return
  fi
  echo "  [statmt] $url → $out"
  mkdir -p "$(dirname "$out")"
  curl -fL "$url" -o "$out"
}

# ─── 1. Europarl-v10 (de-en only; European Parliament) ──────────────────
echo ""
echo "─── [1/4] Europarl-v10 (de-en) ─────────────────────────────────────"
DEEN_DIR="${PARALLEL}/de-en"
mkdir -p "$DEEN_DIR"
if [ -f "${DEEN_DIR}/europarl.de" ] && [ -f "${DEEN_DIR}/europarl.en" ]; then
  echo "  europarl.{de,en} already present — skipping"
else
  fetch_statmt \
    "https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz" \
    "/tmp/europarl-v10.de-en.tsv.gz"
  echo "  extracting (tsv → .de + .en)..."
  gunzip -c /tmp/europarl-v10.de-en.tsv.gz | \
    awk -F'\t' -v de="${DEEN_DIR}/europarl.de" -v en="${DEEN_DIR}/europarl.en" \
    '{print $1 > de; print $2 > en}'
  rm -f /tmp/europarl-v10.de-en.tsv.gz
  echo "    europarl.de: $(wc -l < "${DEEN_DIR}/europarl.de") lines"
  echo "    europarl.en: $(wc -l < "${DEEN_DIR}/europarl.en") lines"
fi

# ─── 2. news-commentary-v16 (de-en, ru-en) ──────────────────────────────
echo ""
echo "─── [2/4] news-commentary-v16 ─────────────────────────────────────"
NC_URL="https://data.statmt.org/news-commentary/v16/training"
for pair in de-en ru-en; do
  src=${pair%-*}; tgt=${pair#*-}
  out_dir="${PARALLEL}/${pair}"
  mkdir -p "$out_dir"
  if [ -f "${out_dir}/news-commentary.${src}" ] && [ -f "${out_dir}/news-commentary.${tgt}" ]; then
    echo "  [$pair] already present — skipping"
    continue
  fi
  tsv="/tmp/nc-v16.${pair}.tsv.gz"
  fetch_statmt "${NC_URL}/news-commentary-v16.${pair}.tsv.gz" "$tsv"
  gunzip -c "$tsv" | \
    awk -F'\t' -v s="${out_dir}/news-commentary.${src}" -v t="${out_dir}/news-commentary.${tgt}" \
    '{print $1 > s; print $2 > t}'
  rm -f "$tsv"
  echo "    news-commentary.${src}: $(wc -l < "${out_dir}/news-commentary.${src}") lines"
done

# ─── 3. TED2020 via OPUS (de-en, ru-en, ar-en) — under parallel_clean/ ──
echo ""
echo "─── [3/4] TED2020 (de-en, ru-en, ar-en) ───────────────────────────"
TED2020_URL="https://object.pouta.csc.fi/OPUS-TED2020/v1/moses"
for pair in de-en ru-en ar-en; do
  src=${pair%-*}; tgt=${pair#*-}
  out_dir="${PARALLEL}/${pair}"
  mkdir -p "$out_dir"
  if [ -f "${out_dir}/ted2020.${src}" ] && [ -f "${out_dir}/ted2020.${tgt}" ]; then
    echo "  [$pair] already present — skipping"
    continue
  fi
  fetch_opus "${TED2020_URL}/${pair}.txt.zip" "/tmp/ted2020_${pair}" \
             "TED2020.${pair}.${src}"
  cp "/tmp/ted2020_${pair}/TED2020.${pair}.${src}" "${out_dir}/ted2020.${src}"
  cp "/tmp/ted2020_${pair}/TED2020.${pair}.${tgt}" "${out_dir}/ted2020.${tgt}"
  rm -rf "/tmp/ted2020_${pair}"
  echo "    ted2020.${src}: $(wc -l < "${out_dir}/ted2020.${src}") lines"
done

# ─── 4. TED2020 en-vi — under raw/ted2020-en-vi/ (different layout) ─────
echo ""
echo "─── [4/4] TED2020 (en-vi) ─────────────────────────────────────────"
VI_DIR="${RAW}/ted2020-en-vi"
mkdir -p "$VI_DIR"
if [ -f "${VI_DIR}/TED2020.en-vi.vi" ] && [ -f "${VI_DIR}/TED2020.en-vi.en" ]; then
  echo "  en-vi already present — skipping"
else
  fetch_opus "${TED2020_URL}/en-vi.txt.zip" "$VI_DIR" "TED2020.en-vi.vi"
  echo "    TED2020.en-vi.vi: $(wc -l < "${VI_DIR}/TED2020.en-vi.vi") lines"
fi

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "  Curated corpus download complete."
echo "  → ${ROOT}"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Next: build the source pool with"
echo "    bin/01_build_source_pool"
echo "  (reads from ${SIMT_CORPUS_ROOT:-\$SIMT_DATA_ROOT}/{parallel_clean,raw}/)"
