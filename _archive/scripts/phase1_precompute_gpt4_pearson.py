"""
Precompute GPT-4 per-sentence Pearson(i*/n, j/m) on the full SiMT-660K corpus,
then stratified-sample 200 sentences into fixed absolute reordering bins.

Runs on the login node — pure chunk arithmetic on already-tokenised text, no GPU.
Uses batched tokenizer calls (process rows in batches of BATCH_SIZE) so total
runtime is ~5-15 min on 660K rows instead of the ~90 min a per-call loop takes.

Approximation: chunk token lengths are computed by tokenising each chunk
independently (add_special_tokens=False) and summing. This ignores the
whitespace-boundary slop between chunks (~1-2 tokens per chunk vs joining and
re-tokenising). Acceptable here because we bin on absolute Pearson thresholds
(0.90, 0.70) and single-token shifts don't move sentences between bins.

Outputs:
  results/gate1/gpt4_pearson_full.json
    { config, summary, per_sentence: { index → { pearson, gpt4_chunks, latency,
      n_src_tok_ours, n_src_tok_gpt4, n_tgt_tok_gpt4, bin } } }
  results/gate1/gate1_indices.json
    { seed, n_per_bin, thresholds, bins: {monotone,mild,reordering}, indices }

Bins (fixed absolute thresholds — advisor rule for cross-run comparability):
  monotone      : GPT-4 Pearson ≥ 0.90
  mild          : 0.70 ≤ Pearson < 0.90
  reordering    : Pearson < 0.70
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT


CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"
BATCH_SIZE = 5000  # rows per tokenizer batch


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return float("nan")
    return sxy / denom


def commit_from_chunk_lens(src_lens, tgt_lens):
    """Build a per-target-token commit trace from per-chunk source/target
    token lengths. Chunk k occupies target positions [prev_tgt, prev_tgt+tgt_lens[k])
    and its commit value is the cumulative source-token count through chunk k."""
    K = len(src_lens)
    cum_src, cum_tgt = 0, 0
    commit = []
    for k in range(K):
        cum_src += src_lens[k]
        for _ in range(tgt_lens[k]):
            commit.append(cum_src)
        cum_tgt += tgt_lens[k]
    if not commit:
        return None, 0, 0
    return _enforce_monotone(commit), cum_src, cum_tgt


def bin_for(pearson_val, mono_thresh, reord_thresh):
    if math.isnan(pearson_val):
        return "undefined"
    if pearson_val >= mono_thresh:
        return "monotone"
    if pearson_val < reord_thresh:
        return "reordering"
    return "mild"


def process_batch(batch_rows, tokenizer, args):
    """Process a list of rows via one batched tokenizer call each for source,
    target, and per-chunk lengths. Returns list of (index, record) tuples for
    kept rows, plus (skipped_len, failed) counts."""
    if not batch_rows:
        return [], 0, 0

    # Batched tokenise the full source strings (for the max_src_tokens filter).
    src_strs = [r["source"] for r in batch_rows]
    src_ids_batch = tokenizer(src_strs, add_special_tokens=False)["input_ids"]

    # Which rows pass the length filter — index into batch.
    keep_flags = [len(ids) <= args.max_src_tokens for ids in src_ids_batch]
    skipped_len = sum(1 for f in keep_flags if not f)
    keep_indices = [k for k, f in enumerate(keep_flags) if f]
    kept_rows = [batch_rows[k] for k in keep_indices]
    kept_src_len = [len(src_ids_batch[k]) for k in keep_indices]

    # Flatten kept sentences' chunk strings for batched tokenisation.
    flat_src_chunks, flat_tgt_chunks = [], []
    chunk_offsets = []  # (start, end) into the flat arrays, per kept row
    for r in kept_rows:
        src_chunks = r["source_chunks"] or []
        tgt_chunks = r["target_chunks"] or []
        if len(src_chunks) != len(tgt_chunks) or not src_chunks:
            chunk_offsets.append((0, 0))
            continue
        chunk_offsets.append((len(flat_src_chunks), len(flat_src_chunks) + len(src_chunks)))
        flat_src_chunks.extend(src_chunks)
        flat_tgt_chunks.extend(tgt_chunks)

    if flat_src_chunks:
        src_chunk_lens_flat = [
            len(ids) for ids in
            tokenizer(flat_src_chunks, add_special_tokens=False)["input_ids"]
        ]
        tgt_chunk_lens_flat = [
            len(ids) for ids in
            tokenizer(flat_tgt_chunks, add_special_tokens=False)["input_ids"]
        ]
    else:
        src_chunk_lens_flat, tgt_chunk_lens_flat = [], []

    out, failed = [], 0
    for i, r in enumerate(kept_rows):
        s, e = chunk_offsets[i]
        if e == s:
            failed += 1
            continue
        src_lens = src_chunk_lens_flat[s:e]
        tgt_lens = tgt_chunk_lens_flat[s:e]
        commit, cum_src, cum_tgt = commit_from_chunk_lens(src_lens, tgt_lens)
        if commit is None or cum_src == 0 or cum_tgt == 0:
            failed += 1
            continue
        m = len(commit)
        xs = [c / cum_src for c in commit]
        ys = [(j + 1) / m for j in range(m)]
        p = pearson(xs, ys)
        b = bin_for(p, args.mono_thresh, args.reord_thresh)
        out.append((r["index"], {
            "pearson": p,
            "gpt4_chunks": len(r["source_chunks"]),
            "latency": r["latency"],
            "n_src_tok_ours": kept_src_len[i],
            "n_src_tok_gpt4": cum_src,
            "n_tgt_tok_gpt4": cum_tgt,
            "bin": b,
        }))

    return out, skipped_len, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", default=str(PRIMARY_BACKBONE))
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--n_per_bin", type=int, default=70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mono_thresh", type=float, default=0.90)
    ap.add_argument("--reord_thresh", type=float, default=0.70)
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "results" / "gate1")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, cap corpus size (for quick smoke). 0 = full 660K.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_path} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Loading corpus from {CORPUS} ...", flush=True)
    t0 = time.time()
    with open(CORPUS) as f:
        rows = json.load(f)
    if args.limit:
        rows = rows[: args.limit]
    print(f"  {len(rows):,} rows in {time.time() - t0:.1f}s", flush=True)

    print(f"Computing GPT-4 per-sentence Pearson (batch={BATCH_SIZE}, "
          f"max_src_tokens={args.max_src_tokens}) ...", flush=True)
    t_start = time.time()
    records = {}
    skipped_len = 0
    failed = 0

    for b_start in range(0, len(rows), BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, len(rows))
        batch = rows[b_start:b_end]
        out, sl, fl = process_batch(batch, tokenizer, args)
        for idx, rec in out:
            records[idx] = rec
        skipped_len += sl
        failed += fl
        elapsed = time.time() - t_start
        done = b_end
        rate = done / elapsed
        eta = (len(rows) - done) / rate if rate > 0 else 0
        print(f"  [{done:>7,}/{len(rows):,}]  rate={rate:.0f} rows/s  "
              f"eta={eta/60:.1f} min  kept={len(records):,}  "
              f"skipped={skipped_len:,}  failed={failed:,}", flush=True)

    total_dt = time.time() - t_start
    print(f"\nComputed for {len(records):,} sentences in {total_dt:.1f}s ({total_dt/60:.1f} min)", flush=True)
    print(f"  skipped (source > {args.max_src_tokens} tok): {skipped_len:,}", flush=True)
    print(f"  failed  (chunk arithmetic):                   {failed:,}", flush=True)

    bin_counter = Counter(v["bin"] for v in records.values())
    print(f"\nBin distribution:", flush=True)
    for b in ["monotone", "mild", "reordering", "undefined"]:
        c = bin_counter.get(b, 0)
        pct = 100 * c / max(len(records), 1)
        print(f"  {b:>12s}: {c:>7,} ({pct:5.1f}%)", flush=True)

    full_path = args.output_dir / "gpt4_pearson_full.json"
    with open(full_path, "w") as f:
        json.dump({
            "config": {
                "corpus": str(CORPUS),
                "tokenizer_path": args.tokenizer_path,
                "max_src_tokens": args.max_src_tokens,
                "mono_thresh": args.mono_thresh,
                "reord_thresh": args.reord_thresh,
                "chunk_len_note": "chunks tokenised independently (whitespace-boundary slop ~1-2 tok/chunk); acceptable for bin thresholds",
            },
            "summary": {
                "n_kept": len(records),
                "n_skipped_len": skipped_len,
                "n_failed": failed,
                "bin_counts": dict(bin_counter),
            },
            "per_sentence": records,
        }, f)
    print(f"\nWrote {full_path}", flush=True)

    # Stratified sample.
    rng = random.Random(args.seed)
    by_bin = {"monotone": [], "mild": [], "reordering": []}
    for idx, rec in records.items():
        if rec["bin"] in by_bin:
            by_bin[rec["bin"]].append(idx)

    picks = {}
    for b, pool in by_bin.items():
        rng.shuffle(pool)
        picks[b] = sorted(pool[: args.n_per_bin])
        print(f"  sampled {len(picks[b])} from {b} bin (pool size {len(pool):,})", flush=True)

    combined = sorted(picks["monotone"] + picks["mild"] + picks["reordering"])

    sample_path = args.output_dir / "gate1_indices.json"
    with open(sample_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_per_bin": args.n_per_bin,
            "thresholds": {"monotone": args.mono_thresh, "reordering": args.reord_thresh},
            "bins": picks,
            "indices": combined,
            "n_total": len(combined),
        }, f, indent=2)
    print(f"\nWrote {sample_path} ({len(combined)} indices)", flush=True)


if __name__ == "__main__":
    main()
