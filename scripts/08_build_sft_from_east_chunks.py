"""Build an SFT dataset from EAST's shipped GPT-4 chunks (SiMT-Multi-90K).

Stage 3 alternative to 08_build_sft_dataset.py: instead of deriving chunk
boundaries from an annotator's divergence matrices, take the boundaries
Multi-90K already ships and turn them into the token-id form src/train/sft.py
consumes.

Boundaries are snapped by character offset, so the training tokenization is
byte-identical to what streaming inference produces at test time. A row is
dropped if a chunk string cannot be located in its sentence or a boundary
falls inside a token.

Adapted from _archive/scripts/phase2_build_condA_dataset.py, with the
source-pool filter made optional so the whole corpus can be used.

Usage:
    python scripts/08_build_sft_from_east_chunks.py \\
        --multi90k ${SIMT_DATA_ROOT}/SiMT-Multi-90K/SiMT-Multi-90K.json \\
        --tokenizer_path results/train/gemma_2b_m90k/tokenizer \\
        --directions de-en en-de ru-en en-ru \\
        --output results/sft_dataset/gemma_2b_m90k/sft_dataset.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LANG_TO_CODE = {"German": "de", "English": "en", "Russian": "ru",
                "Chinese": "zh", "Czech": "cs"}


def snap_chunks_to_tokens(text, chunks, tokenizer):
    # Find each chunk's character span, then cut the sentence's token ids at
    # those spans. Returns None if a boundary does not line up.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc.input_ids, enc.offset_mapping

    boundaries, cursor = [], 0
    for chunk in chunks:
        stripped = chunk.strip()
        start = text.find(stripped, cursor)
        if start < 0:
            return None
        cursor = start + len(stripped)
        boundaries.append(cursor)

    per_chunk, tok_start = [], 0
    for boundary in boundaries:
        tok_end = tok_start
        while tok_end < len(offsets) and offsets[tok_end][1] <= boundary:
            tok_end += 1
        per_chunk.append(ids[tok_start:tok_end])
        tok_start = tok_end
    # Anything left over belongs to the final chunk.
    if tok_start < len(ids):
        if not per_chunk:
            return None
        per_chunk[-1] = per_chunk[-1] + ids[tok_start:]
    if any(len(c) == 0 for c in per_chunk):
        return None
    return per_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi90k", type=Path, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--directions", nargs="+",
                    default=["de-en", "en-de", "ru-en", "en-ru"])
    ap.add_argument("--corpus_json", type=Path, default=None,
                    help="optional source pool; keeps only rows whose source "
                         "string appears in it, for a seed-matched comparison")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rows = json.loads(args.multi90k.read_text(encoding="utf-8"))
    print(f"Multi-90K rows: {len(rows):,}", flush=True)

    # A source sentence ships once per latency label, so index by direction
    # and source string and keep every variant.
    by_source = defaultdict(list)
    for r in rows:
        src = LANG_TO_CODE.get(r["src_lang"])
        tgt = LANG_TO_CODE.get(r["tgt_lang"])
        if src is None or tgt is None:
            continue
        by_source[(f"{src}-{tgt}", r["source"])].append(r)

    wanted = None
    if args.corpus_json:
        pool = json.loads(args.corpus_json.read_text(encoding="utf-8"))
        wanted = {(p["direction"], p["source"]) for p in pool}
        print(f"Source pool: {len(wanted):,} (direction, source) keys", flush=True)

    kept, stats = [], {d: Counter() for d in args.directions}
    for (direction, source), variants in by_source.items():
        if direction not in args.directions:
            continue
        if wanted is not None and (direction, source) not in wanted:
            continue
        stats[direction]["sources"] += 1
        for m in variants:
            src_chunks, tgt_chunks = m["source_chunks"], m["target_chunks"]
            if not src_chunks or len(src_chunks) != len(tgt_chunks):
                stats[direction]["bad_chunk_count"] += 1
                continue
            src_ids = snap_chunks_to_tokens(m["source"], src_chunks, tokenizer)
            tgt_ids = snap_chunks_to_tokens(m["target"], tgt_chunks, tokenizer)
            if src_ids is None or tgt_ids is None:
                stats[direction]["snap_failed"] += 1
                continue
            kept.append({
                "index": len(kept),
                "source": m["source"],
                "target": m["target"],
                "src_lang": m["src_lang"],
                "tgt_lang": m["tgt_lang"],
                "latency": m["latency"],
                "source_chunks": src_chunks,
                "target_chunks": tgt_chunks,
                "source_chunk_ids": src_ids,
                "target_chunk_ids": tgt_ids,
                "_annotator_meta": {"chunks_source": "GPT-4 (SiMT-Multi-90K)",
                                    "augmented_from_base": False},
            })
            stats[direction]["kept"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")

    print("\nPer direction:")
    for d in args.directions:
        s = stats[d]
        print(f"  {d}: sources={s['sources']:,}  kept={s['kept']:,}  "
              f"snap_failed={s['snap_failed']}  bad_chunk_count={s['bad_chunk_count']}")
    print(f"\nTotal rows: {len(kept):,}")

    lat = Counter(r["latency"] for r in kept)
    print("\nLatency distribution:")
    for name in ["low", "medium", "high"]:
        print(f"  {name:>6}: {lat.get(name, 0):>6,} ({lat.get(name, 0) * 100 / max(len(kept), 1):.1f}%)")

    chunks = Counter(len(r["source_chunks"]) for r in kept)
    print("\nChunks per sentence (top 8):")
    for count, n in sorted(chunks.items())[:8]:
        print(f"  {count:>3d}: {n:,} rows")


if __name__ == "__main__":
    main()
