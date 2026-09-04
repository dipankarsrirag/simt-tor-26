#!/usr/bin/env bash
# Build the eval test sets in the layout src/eval/extrinsic.py reads:
#
#   ${SIMT_TESTSETS_ROOT}/eval/{src-tgt}/{set}.{src-tgt}.src
#   ${SIMT_TESTSETS_ROOT}/eval/{src-tgt}/{set}.{src-tgt}.ref
#
# WMT15, WMT22 and IWSLT17 come from sacrebleu. IWSLT15 en-vi is not in
# sacrebleu, so it is read from a parquet copy of the tst2013 split on the Hub
# (1,268 lines, Stanford tokenisation, spaces before punctuation).
#
# Idempotent: a pair already present is skipped.
#
# Usage:
#   bash scripts/fetch_eval_testsets.sh

set -uo pipefail

EVAL_ROOT="${SIMT_TESTSETS_ROOT:-${SIMT_DATA_ROOT:-$(pwd)/data}}/eval"
PYTHON="${PYTHON:-python3}"
mkdir -p "$EVAL_ROOT"

# One direction of one sacrebleu test set.
fetch () {
  local set_name=$1 sacrebleu_name=$2 pair=$3
  local dir="${EVAL_ROOT}/${pair}"
  mkdir -p "$dir"
  if [ -s "${dir}/${set_name}.${pair}.src" ]; then
    echo "  [$set_name $pair] already present"
    return
  fi
  if sacrebleu -t "$sacrebleu_name" -l "$pair" --echo src > "${dir}/${set_name}.${pair}.src" 2>/dev/null \
     && sacrebleu -t "$sacrebleu_name" -l "$pair" --echo ref > "${dir}/${set_name}.${pair}.ref" 2>/dev/null; then
    echo "  [$set_name $pair] $(wc -l < "${dir}/${set_name}.${pair}.src") lines"
  else
    echo "  [$set_name $pair] FAILED"
    rm -f "${dir}/${set_name}.${pair}.src" "${dir}/${set_name}.${pair}.ref"
  fi
}

echo "=== WMT15 + WMT22 + IWSLT17 (sacrebleu) -> ${EVAL_ROOT}"
fetch wmt15 wmt15 de-en
for pair in de-en en-de ru-en en-ru; do fetch wmt22 wmt22 "$pair"; done
for pair in de-en en-de ar-en en-ar; do fetch iwslt17 iwslt17 "$pair"; done

echo "=== IWSLT15 en-vi tst2013 (Hub parquet)"
if [ -s "${EVAL_ROOT}/en-vi/iwslt15.en-vi.src" ]; then
  echo "  [iwslt15 en-vi] already present"
else
  "$PYTHON" - "$EVAL_ROOT" <<'PY'
import os, sys
from datasets import load_dataset
eval_root = sys.argv[1]
ds = load_dataset("thainq107/iwslt2015-en-vi", split="test")
en = [r["en"].strip().replace("\n", " ") for r in ds]
vi = [r["vi"].strip().replace("\n", " ") for r in ds]
for pair, src, ref in [("en-vi", en, vi), ("vi-en", vi, en)]:
    d = os.path.join(eval_root, pair)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"iwslt15.{pair}.src"), "w", encoding="utf-8").write("\n".join(src) + "\n")
    open(os.path.join(d, f"iwslt15.{pair}.ref"), "w", encoding="utf-8").write("\n".join(ref) + "\n")
print(f"  [iwslt15] {len(en)} lines per direction")
PY
fi

echo "=== done"
wc -l "${EVAL_ROOT}"/*/*.src 2>/dev/null | tail -1
