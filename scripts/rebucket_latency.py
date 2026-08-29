"""Relabel latency on an already-built SFT dataset in place.

Delegates to `latency_from_chunk_stats` in phase2_build_sft_dataset.py — the
single source of truth for the latency rule. Use this to retrofit an old
dataset (e.g. one built before 2026-08-22 with chunk-count-only thresholds)
without re-running the annotator.

For NEW dataset builds, `phase2_build_sft_dataset.py` already uses the same
rule by default (via `latency_from_chunk_stats`); no post-hoc rebucketing
step is needed.

Reads:  arbitrary sft_dataset_*.json (list of rows with source, src_lang,
        source_chunks fields)
Writes: same schema with `latency` overwritten and prior label stored in
        `_annotator_meta.rebucket_rule.prior_latency`
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase2_build_sft_dataset import (
    LATENCY_CC1_MAX, LATENCY_LOW_CCSW, LATENCY_MED_CCSW,
    _count_source_words, latency_from_chunk_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"Reading {args.input}", flush=True)
    with open(args.input) as f:
        rows = json.load(f)
    print(f"  n={len(rows):,}", flush=True)
    print(f"  rule: cc<={LATENCY_CC1_MAX} -> high; else cc/sw>={LATENCY_LOW_CCSW} low, "
          f">={LATENCY_MED_CCSW} medium, else high", flush=True)

    flip = Counter()
    n_changed = 0
    for r in rows:
        sw = _count_source_words(r["source"], r.get("src_lang", "en"))
        cc = len(r.get("source_chunks") or [])
        old = r["latency"]
        new = latency_from_chunk_stats(cc, sw)
        if new != old:
            n_changed += 1
        flip[(old, new)] += 1
        r["latency"] = new
        meta = r.setdefault("_annotator_meta", {})
        meta["rebucket_rule"] = {
            "cc1_threshold": LATENCY_CC1_MAX,
            "low_threshold": LATENCY_LOW_CCSW,
            "medium_threshold": LATENCY_MED_CCSW,
            "prior_latency": old,
        }

    print(f"\nRelabelled: {n_changed:,} / {len(rows):,} rows ({n_changed*100/len(rows):.1f}%)", flush=True)
    print(f"\nFlip table (old -> new):", flush=True)
    for (old, new), c in sorted(flip.items(), key=lambda x: -x[1]):
        arrow = "  " if old == new else "->"
        print(f"  {old:>7s} {arrow}{new:>7s}   {c:>7,}", flush=True)

    marg = Counter(r["latency"] for r in rows)
    n = len(rows)
    print(f"\nNew marginal:", flush=True)
    for lab in ("low", "medium", "high"):
        print(f"  {lab:>7s}: {marg[lab]:>7,}  ({marg[lab]*100/n:>5.1f}%)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"\nWrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
