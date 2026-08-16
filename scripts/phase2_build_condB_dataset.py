"""
Build the cond-B SFT training corpus from our OT annotator's matrices.

Reads:
  results/phase2/annot_ot_condB_n2k/matrices.jsonl
Writes:
  results/phase2/condB_n2k_dataset.json  — same schema as SiMT-660K.json
                                            but with our-annotator chunks

The output can be fed to `src/train/sft.py --corpus_file <path>`.

Chunk derivation. For each sentence:
  1. Load the (n, m) divergence matrix.
  2. Commit at a chosen tau per METHOD §4 (commit_from_matrix + enforce_monotone).
  3. Group consecutive commit points into (source_chunk, target_chunk) pairs via
     _chunks_from_commit (same routine used by the annotator online).

τ strategy — start with the tightest fixed-τ policy that avoids collapse.
Gate-1 Config F used τ=0.30 as the primary; that's what we ship as default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _chunks_from_commit, _enforce_monotone
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def commit_from_matrix(matrix, tau, n):
    if not matrix or not matrix[0]:
        return []
    m = len(matrix[0])
    commit = [n] * m
    for i_row, row in enumerate(matrix):
        i = i_row + 1
        for j in range(m):
            if commit[j] < n:
                continue
            d = row[j]
            if not math.isnan(d) and d < tau:
                commit[j] = i
    return _enforce_monotone(commit)


def chunk_count(commit):
    if not commit:
        return 0
    m = len(commit)
    g, j = 0, 0
    while j < m:
        k = j
        while k + 1 < m and commit[k + 1] == commit[j]:
            k += 1
        g += 1
        j = k + 1
    return g


def build_dataset(matrices_path: Path, tau: float, tokenizer, corpus_by_idx: dict,
                  collapse_policy: str = "keep"):
    """Return list of dicts matching SiMT-660K.json schema. `collapse_policy`:
    'keep' — keep single-chunk collapses as valid training rows (their chunk
             becomes the whole source + whole target — highest-latency case);
    'skip' — drop them.
    """
    from src.annotator.east_format import EastRow, interleave

    kept, skipped = [], 0
    n_collapse = 0
    n_missing = 0

    with open(matrices_path) as f:
        for line in f:
            rec = json.loads(line)
            idx = rec["index"]
            if idx not in corpus_by_idx:
                n_missing += 1
                continue
            src_row = corpus_by_idx[idx]
            n = rec["n_src_tok"]
            m = rec["n_tgt_tok"]
            commit = commit_from_matrix(rec["matrix"], tau, n)
            cc = chunk_count(commit)

            if cc == 1:
                n_collapse += 1
                if collapse_policy == "skip":
                    skipped += 1
                    continue
            # Tokenise the source and target to get token id lists.
            src_ids = tokenizer(src_row["source"], add_special_tokens=False)["input_ids"]
            tgt_ids = tokenizer(src_row["target"], add_special_tokens=False)["input_ids"]
            # commit was computed against a matrix of shape (n_src_tok, n_tgt_tok)
            # which is len(src_ids), len(tgt_ids). But the annotator recorded these.
            # Confirm they still line up (drop otherwise).
            if len(src_ids) != n or len(tgt_ids) != m:
                skipped += 1
                continue
            source_chunks, target_chunks = _chunks_from_commit(
                commit, src_ids, tgt_ids, tokenizer, n
            )
            if len(source_chunks) != len(target_chunks) or not source_chunks:
                skipped += 1
                continue
            # Sanity: verify interleave doesn't raise.
            try:
                _ = interleave(EastRow(
                    source=src_row["source"], target=src_row["target"],
                    src_lang=src_row["src_lang"], tgt_lang=src_row["tgt_lang"],
                    latency=src_row["latency"],
                    source_chunks=source_chunks, target_chunks=target_chunks,
                ))
            except Exception:
                skipped += 1
                continue

            kept.append({
                "index": idx,
                "source": src_row["source"],
                "target": src_row["target"],
                "src_lang": src_row["src_lang"],
                "tgt_lang": src_row["tgt_lang"],
                "latency": src_row["latency"],
                "source_chunks": source_chunks,
                "target_chunks": target_chunks,
            })

    return kept, {"missing": n_missing, "skipped": skipped, "collapsed": n_collapse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "annot_ot_condB_n2k" / "matrices.jsonl")
    ap.add_argument("--tokenizer_path", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--tau", type=float, default=0.30,
                    help="OT divergence threshold. Default 0.30 (Gate-1 Config F primary).")
    ap.add_argument("--collapse_policy", choices=["keep", "skip"], default="keep",
                    help="Single-chunk-collapse rows: keep as high-latency training "
                         "examples (default) or skip. 'keep' is the honest choice — "
                         "these are the true late-commit cases where our tags say "
                         "'read whole source then emit all'.")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "condB_n2k_dataset.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading base corpus from {CORPUS}", flush=True)
    with open(CORPUS) as f:
        corpus_rows = json.load(f)
    corpus_by_idx = {r["index"]: r for r in corpus_rows}
    print(f"  {len(corpus_by_idx):,} rows indexed", flush=True)

    print(f"Building cond-B dataset from {args.matrices} at tau={args.tau} "
          f"(collapse={args.collapse_policy}) ...", flush=True)
    kept, stats = build_dataset(args.matrices, args.tau, tokenizer, corpus_by_idx,
                                collapse_policy=args.collapse_policy)
    print(f"  kept={len(kept)}  skipped={stats['skipped']}  "
          f"missing={stats['missing']}  collapsed={stats['collapsed']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)", flush=True)

    # Diagnostic — a sample of chunk counts.
    from collections import Counter
    cc_dist = Counter(len(r["source_chunks"]) for r in kept)
    print(f"\nChunk-count distribution (top 10):", flush=True)
    for c, n in cc_dist.most_common(10):
        print(f"  {c:>3d} chunks: {n} rows", flush=True)


if __name__ == "__main__":
    main()
