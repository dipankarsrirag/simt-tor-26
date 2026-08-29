"""Build cond-A (GPT-4 chunks) dataset from Multi-90K, matched to v6b seed.

For a fair head-to-head vs v6b-ctrl, cond-A uses:
  - Same source/target texts as v6b (looked up from Multi-90K by source string)
  - Multi-90K's shipped GPT-4 source_chunks/target_chunks
  - Position-based slicing (tokenize full source once, then snap each GPT
    chunk boundary to the nearest token position) — matches cond-B's
    direct-ids splice pipeline, so training tokenization is byte-identical
    to what streaming inference produces at test time.
  - Rows dropped when: no source-string match in Multi-90K OR chunk boundary
    can't be snapped to a token position (rare)

Directions covered: de-en, en-de, ru-en, en-ru (the 4 v6b directions that
Multi-90K covers). ar/vi excluded since Multi-90K doesn't ship those.

Output: results/phase2/sft_dataset_multilingual_v6b_condA.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _is_cjk_lang

POOL = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/multilingual_source_pool_v5.json")
MULTI90K = Path("/g/data/ba39/dipankar/simt-tor-26/data/SiMT-Multi-90K/SiMT-Multi-90K.json")
OUT = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/sft_dataset_multilingual_v6b_condA.json")
TOKENIZER = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended-v6"

TARGET_DIRS = ["de-en", "en-de", "ru-en", "en-ru"]
LANG_TO_FULL = {"de": "German", "en": "English", "ru": "Russian",
                "zh": "Chinese", "cs": "Czech"}
FULL_TO_CODE = {v: k for k, v in LANG_TO_FULL.items()}


def snap_chunks_to_tokens(source: str, source_chunks: list[str], tokenizer):
    """Given a full source string and a list of GPT chunk strings, return
    per-chunk token-id lists such that concatenated == tok(source).

    Uses character offsets: walk chunks in order, find each chunk's char span
    in source, then use tokenizer's offset_mapping to slice.

    Returns None if any chunk boundary can't be snapped cleanly.
    """
    full_ids_enc = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = full_ids_enc.input_ids
    offsets = full_ids_enc.offset_mapping

    chunk_boundaries_chars = []
    cursor = 0
    for chunk in source_chunks:
        c_stripped = chunk.strip()
        idx = source.find(c_stripped, cursor)
        if idx < 0:
            return None
        end = idx + len(c_stripped)
        chunk_boundaries_chars.append(end)
        cursor = end

    per_chunk_ids = []
    tok_start = 0
    for boundary_char in chunk_boundaries_chars:
        tok_end = tok_start
        while tok_end < len(offsets) and offsets[tok_end][1] <= boundary_char:
            tok_end += 1
        per_chunk_ids.append(full_ids[tok_start:tok_end])
        tok_start = tok_end
    if tok_start < len(full_ids):
        if per_chunk_ids:
            per_chunk_ids[-1] = per_chunk_ids[-1] + full_ids[tok_start:]
        else:
            return None
    if any(len(c) == 0 for c in per_chunk_ids):
        return None
    return per_chunk_ids


def main():
    from transformers import AutoTokenizer
    print(f"Loading tokenizer {TOKENIZER} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    print("Loading pool ...", flush=True)
    with open(POOL) as f:
        pool = json.load(f)
    print("Loading Multi-90K ...", flush=True)
    with open(MULTI90K) as f:
        m90 = json.load(f)

    # Index Multi-90K by (direction, source) -> LIST of rows (not single row).
    # Each source in Multi-90K appears at up to 3 latency variants; the prior
    # `m90_lookup[key] = row` overwrote earlier variants, silently keeping
    # only the LAST one (which is always "high" per EAST's file ordering).
    # See LOG.md 2026-08-23 for the diagnostic.
    from collections import defaultdict
    m90_lookup = defaultdict(list)
    for r in m90:
        src_code = FULL_TO_CODE.get(r["src_lang"])
        tgt_code = FULL_TO_CODE.get(r["tgt_lang"])
        if src_code is None or tgt_code is None:
            continue
        m90_lookup[(f"{src_code}-{tgt_code}", r["source"])].append(r)

    total_variants = sum(len(v) for v in m90_lookup.values())
    print(f"Multi-90K indexed: {len(m90_lookup)} unique (dir, source) tuples across "
          f"{total_variants} rows", flush=True)

    kept = []
    stats = {d: {"pool": 0, "matched": 0, "snap_ok": 0, "final": 0} for d in TARGET_DIRS}
    for r in pool:
        d = r["direction"]
        if d not in TARGET_DIRS:
            continue
        stats[d]["pool"] += 1
        matches = m90_lookup.get((d, r["source"]))
        if not matches:
            continue
        stats[d]["matched"] += 1
        # Emit ALL latency variants of this source (EAST's design ships up to 3)
        for m in matches:
            src_chunks_str = m["source_chunks"]
            tgt_chunks_str = m["target_chunks"]
            if not src_chunks_str or len(src_chunks_str) != len(tgt_chunks_str):
                continue
            src_chunk_ids = snap_chunks_to_tokens(m["source"], src_chunks_str, tok)
            tgt_chunk_ids = snap_chunks_to_tokens(m["target"], tgt_chunks_str, tok)
            if src_chunk_ids is None or tgt_chunk_ids is None:
                continue
            stats[d]["snap_ok"] += 1
            kept.append({
                "index": r["index"],
                "source": m["source"],
                "target": m["target"],
                "src_lang": d.split("-")[0],
                "tgt_lang": d.split("-")[1],
                "latency": m["latency"],
                "source_chunks": src_chunks_str,
                "target_chunks": tgt_chunks_str,
                "source_chunk_ids": src_chunk_ids,
                "target_chunk_ids": tgt_chunk_ids,
                "_annotator_meta": {
                    "chunks_source": "GPT-4 (Multi-90K)",
                    "augmented_from_base": False,
                    "m90_latency_variant": m["latency"],
                },
            })
            stats[d]["final"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept, ensure_ascii=False))
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)", flush=True)
    print(f"\nPer-direction:")
    for d in TARGET_DIRS:
        s = stats[d]
        print(f"  {d}: pool={s['pool']}  matched={s['matched']}  snap_ok={s['snap_ok']}  final={s['final']}",
              flush=True)
    print(f"\nTotal kept: {len(kept)}", flush=True)

    # Latency distribution
    from collections import Counter
    lat_dist = Counter(r["latency"] for r in kept)
    print(f"\nLatency distribution:")
    for lat in ["low", "medium", "high"]:
        print(f"  {lat:>6}: {lat_dist.get(lat, 0):>5d} ({lat_dist.get(lat,0)*100/max(len(kept),1):.1f}%)")
    # Chunk-count distribution
    cc = Counter(len(r["source_chunks"]) for r in kept)
    print(f"\nChunk-count distribution (top 10):")
    for c, cnt in sorted(cc.items())[:10]:
        print(f"  {c:>3d} chunks: {cnt} rows")


if __name__ == "__main__":
    main()
