"""
Build the cond-B SFT training corpus from our OT annotator's matrices.

Reads:
  results/phase2/annot_ot_n2k/matrices.jsonl
Writes:
  results/phase2/sft_dataset_n2k.json  — same schema as SiMT-660K.json
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

from src.annotator.annotate import _chunks_from_commit, _enforce_monotone, _is_cjk_lang
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def _chunk_word_count(chunk_str: str, is_cjk: bool) -> int:
    """Count 'words' in a chunk per EAST convention: char count for CJK, split for others."""
    if is_cjk:
        return sum(1 for ch in chunk_str if not ch.isspace())
    return len(chunk_str.split())


def merge_small_chunks(source_chunks, target_chunks, source_chunk_ids, target_chunk_ids,
                        src_lang, tgt_lang,
                        min_src_words=2, min_src_chars_cjk=4):
    """EAST §3.1 merge rule: any source chunk with < `min_src_words` (or
    < `min_src_chars_cjk` characters for CJK sources) is merged into the
    SUBSEQUENT chunk. Both source and target chunks (and their id lists)
    are merged in lock-step to preserve alignment. If the last chunk is
    too small, it is merged into the previous chunk instead.

    Returns (source_chunks, target_chunks, source_chunk_ids, target_chunk_ids),
    each possibly shorter than input.
    """
    src_is_cjk = _is_cjk_lang(src_lang)
    tgt_is_cjk = _is_cjk_lang(tgt_lang)
    src_sep = "" if src_is_cjk else " "
    tgt_sep = "" if tgt_is_cjk else " "
    threshold = min_src_chars_cjk if src_is_cjk else min_src_words

    # First pass: forward-merge (merge small chunk into NEXT). Do it iteratively.
    changed = True
    while changed and len(source_chunks) > 1:
        changed = False
        for i in range(len(source_chunks) - 1):
            if _chunk_word_count(source_chunks[i], src_is_cjk) < threshold:
                # Merge chunk i into chunk i+1
                source_chunks[i + 1] = (source_chunks[i] + src_sep + source_chunks[i + 1]).strip()
                target_chunks[i + 1] = (target_chunks[i] + tgt_sep + target_chunks[i + 1]).strip()
                source_chunk_ids[i + 1] = source_chunk_ids[i] + source_chunk_ids[i + 1]
                target_chunk_ids[i + 1] = target_chunk_ids[i] + target_chunk_ids[i + 1]
                del source_chunks[i]
                del target_chunks[i]
                del source_chunk_ids[i]
                del target_chunk_ids[i]
                changed = True
                break

    # Second pass: if LAST chunk is too small, merge into previous
    if len(source_chunks) > 1 and _chunk_word_count(source_chunks[-1], src_is_cjk) < threshold:
        source_chunks[-2] = (source_chunks[-2] + src_sep + source_chunks[-1]).strip()
        target_chunks[-2] = (target_chunks[-2] + tgt_sep + target_chunks[-1]).strip()
        source_chunk_ids[-2] = source_chunk_ids[-2] + source_chunk_ids[-1]
        target_chunk_ids[-2] = target_chunk_ids[-2] + target_chunk_ids[-1]
        del source_chunks[-1]
        del target_chunks[-1]
        del source_chunk_ids[-1]
        del target_chunk_ids[-1]

    return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids


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


def commit_with_fallback(matrix, tau_ladder, n):
    """Try tau values in order; return the FIRST commit vector with chunks > 1.
    Fallback for the collapse case where the primary tau produces chunks == 1
    (e.g., first target position never converges below tau, and monotone
    enforcement propagates that to all subsequent targets).

    tau_ladder: list of taus to try, e.g. [0.30, 0.50, 0.70, 1.00].
    Returns (commit, tau_used, was_collapsed_at_primary_tau).
    """
    primary_tau = tau_ladder[0]
    for tau in tau_ladder:
        commit = commit_from_matrix(matrix, tau, n)
        if chunk_count(commit) > 1:
            return commit, tau, tau != primary_tau
    # No tau in ladder produced >1 chunks — return commit at largest tau anyway.
    return commit, tau_ladder[-1], True


# Latency thresholds recalibrated 2026-08-19 (v4 rebuild) to match the
# empirical tertiles of our OT-annotator's chunk-count distribution.
# The v2 build distribution (before augmentation) was 27% high / 21% medium
# / 51% low — over-representing low. New bounds shift the medium band up:
#   <= 3 chunks   -> high    (conservative commit; ~27% of raw rows)
#   4 - 6 chunks  -> medium  (was 4-5; adds chunk-count 6 to medium: ~30%)
#   >= 7 chunks   -> low     (was >=6; ~43%)
# Then augmentation (see augment_row_at_lower_chunk_counts) further balances
# the buckets by generating higher-latency versions from many-chunk sources.
LATENCY_HIGH_MAX_CHUNKS = 3    # <= 3 chunks -> high latency
LATENCY_MEDIUM_MAX_CHUNKS = 6  # 4-6 chunks -> medium
                               # >= 7 chunks -> low latency


def latency_from_chunk_count(cc: int) -> str:
    if cc <= LATENCY_HIGH_MAX_CHUNKS:
        return "high"
    if cc <= LATENCY_MEDIUM_MAX_CHUNKS:
        return "medium"
    return "low"


def merge_chunks_to_n(source_chunks, target_chunks, source_chunk_ids, target_chunk_ids,
                       target_n: int, src_lang: str = "en", tgt_lang: str = "en"):
    """Merge consecutive read/write chunks to reduce count to `target_n` chunks.

    Used for latency-augmentation (2026-08-19 v4 rebuild): a low-latency row
    with 8 chunks can be merged into a 4-chunk row (medium) and a 2-chunk row
    (high), giving the model direct exposure to the same source content at
    multiple latency labels.

    Merges are contiguous: group N/target_n consecutive chunks into one.
    Both string and BPE-id representations are updated in lock-step so
    downstream consumers (string-interleave OR direct-ids) both work.

    For CJK sides (zh/ja/ko/th/km), chunks are joined with EMPTY separator
    (no whitespace in those scripts); for others, single-space separator.

    Returns (source_chunks, target_chunks, source_chunk_ids, target_chunk_ids)
    with exactly target_n entries.
    """
    k = len(source_chunks)
    if target_n >= k or target_n <= 0:
        return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids
    src_sep = "" if _is_cjk_lang(src_lang) else " "
    tgt_sep = "" if _is_cjk_lang(tgt_lang) else " "
    per_group = k / target_n
    cuts = [int(round(i * per_group)) for i in range(target_n + 1)]
    cuts[0] = 0
    cuts[-1] = k
    new_src, new_tgt, new_src_ids, new_tgt_ids = [], [], [], []
    for gi in range(target_n):
        s, e = cuts[gi], cuts[gi + 1]
        if s >= e:
            continue
        new_src.append(src_sep.join(c.strip() for c in source_chunks[s:e]).strip())
        new_tgt.append(tgt_sep.join(c.strip() for c in target_chunks[s:e]).strip())
        merged_src_ids = [t for src_id_list in source_chunk_ids[s:e] for t in src_id_list]
        merged_tgt_ids = [t for tgt_id_list in target_chunk_ids[s:e] for t in tgt_id_list]
        new_src_ids.append(merged_src_ids)
        new_tgt_ids.append(merged_tgt_ids)
    return new_src, new_tgt, new_src_ids, new_tgt_ids


def augment_row_at_lower_chunk_counts(row: dict):
    """Given a base row (dict with source_chunks/target_chunks/*_chunk_ids/etc),
    produce augmentation rows at COARSER chunk granularities to expose the
    same source at higher latency labels. Returns a list of NEW rows (does NOT
    include the base row — caller handles that).

    Rule:
      - k >= 4  -> merge to ceil(k/2) chunks    (one coarser step)
      - k >= 7  -> also merge to ceil(k/4) chunks (two coarser steps)
    New rows inherit index (with `_aug` suffix), source/target strings, but
    have merged chunks and a freshly-assigned latency label per new chunk count.
    """
    import copy
    k = len(row["source_chunks"])
    if k < 4:
        return []
    out = []
    for divisor, tag in [(2, "aug2"), (4, "aug4")]:
        target_n = max(1, -(-k // divisor))  # ceil-div
        if target_n >= k:
            continue
        if divisor == 4 and k < 7:
            continue
        new_src, new_tgt, new_src_ids, new_tgt_ids = merge_chunks_to_n(
            row["source_chunks"], row["target_chunks"],
            row["source_chunk_ids"], row["target_chunk_ids"],
            target_n,
            src_lang=row.get("src_lang", "en"),
            tgt_lang=row.get("tgt_lang", "en"),
        )
        if len(new_src) < 1:
            continue
        new_latency = latency_from_chunk_count(len(new_src))
        if new_latency == row["latency"]:
            continue  # no informational gain if latency label doesn't flip
        aug = copy.copy(row)
        aug["source_chunks"] = new_src
        aug["target_chunks"] = new_tgt
        aug["source_chunk_ids"] = new_src_ids
        aug["target_chunk_ids"] = new_tgt_ids
        aug["latency"] = new_latency
        aug_meta = dict(row.get("_annotator_meta", {}))
        aug_meta["augmented_from_base"] = True
        aug_meta["base_n_chunks"] = k
        aug_meta["merged_to_n_chunks"] = len(new_src)
        aug_meta["merge_tag"] = tag
        aug["_annotator_meta"] = aug_meta
        out.append(aug)
    return out


def build_dataset(matrices_path: Path, tau_ladder: list[float], tokenizer,
                  corpus_by_idx: dict, reassign_latency: bool = True,
                  merge_small: bool = False, min_src_words: int = 2,
                  min_src_chars_cjk: int = 4):
    """Return list of dicts matching SiMT-660K.json schema.

    `tau_ladder`: list of tau values, tried in order per row. First tau that
    produces > 1 chunks is used (collapse fallback, 2026-08-18 fix). Primary
    tau is tau_ladder[0]; the rest are fallbacks.
    `reassign_latency`: if True (default), overwrite each row's latency
    label based on our chunk count using EAST-inherited thresholds
    (`latency_from_chunk_count`). If False, inherit SiMT-660K's original
    latency label (which was GPT-4-derived and may be inconsistent with our
    chunks — pre-2026-08-18 behavior).
    """
    from src.annotator.east_format import EastRow, interleave

    kept, skipped = [], 0
    n_collapse_at_primary = 0  # primary tau produced chunks==1; had to fall back
    n_still_collapse = 0       # even fallback ladder failed (rare)
    n_missing = 0
    n_relabelled = 0           # rows where reassigned latency != inherited
    latency_flips = {}         # (old, new) -> count

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

            # Fix 1 (2026-08-18): fallback tau ladder to escape collapse.
            commit, tau_used, fell_back = commit_with_fallback(rec["matrix"], tau_ladder, n)
            cc = chunk_count(commit)
            if fell_back:
                n_collapse_at_primary += 1
            if cc == 1:
                # Even the largest tau in the ladder collapsed. Very rare
                # (target j=0 doesn't converge even at tau_max). Drop.
                n_still_collapse += 1
                skipped += 1
                continue

            src_clean = src_row["source"].strip()
            tgt_clean = src_row["target"].strip()

            # 2026-08-22 v6b fix: use the annotator's ORIGINAL tokenization
            # (no leading space — how it fed the divergence matrix). The
            # v4/v5 "prepend leading space and re-tokenize" gate was a
            # streaming-alignment kludge for v1-v5 (which fed raw source
            # after `<|latency|>` in the prompt — leading space imputed by
            # the string join). v6 goes to direct-ids splice at both
            # training and inference: `src/train/sft_v6.py` builds
            # input_ids from `source_chunk_ids`/`target_chunk_ids` byte-
            # exact (no string round-trip), and streaming's
            # `tokenize_source_by_words` matches by tokenizing word[0]
            # WITHOUT leading space and word[i>0] WITH leading space.
            # The leading-space retokenization gate silently dropped
            # 40-47% of AR/VI rows on v6 (leading space changes segmentation
            # boundaries in RTL/no-Latin scripts). Removing it is
            # correctness-preserving so long as (a) the training pipeline
            # consumes chunk_ids directly and (b) the streaming inference
            # tokenizes per-word to match the annotator's full-source
            # tokenization by concatenation. Both are now the case.
            src_ids_orig = tokenizer(src_clean, add_special_tokens=False)["input_ids"]
            tgt_ids_orig = tokenizer(tgt_clean, add_special_tokens=False)["input_ids"]
            if len(src_ids_orig) != n or len(tgt_ids_orig) != m:
                skipped += 1
                continue

            source_chunks, target_chunks, source_chunk_ids, target_chunk_ids = _chunks_from_commit(
                commit, src_ids_orig, tgt_ids_orig, tokenizer, n, src_lang=src_row["src_lang"]
            )
            if len(source_chunks) != len(target_chunks) or not source_chunks:
                skipped += 1
                continue

            # 2026-08-22: EAST §3.1 merge rule — merge any chunk with < 2 source
            # words (or < 4 chars for CJK) into the subsequent chunk. Brings OT
            # chunks closer to GPT-4's semantic-unit granularity while preserving
            # commit-point-derived adaptive structure.
            if merge_small and len(source_chunks) > 1:
                source_chunks, target_chunks, source_chunk_ids, target_chunk_ids = merge_small_chunks(
                    source_chunks, target_chunks, source_chunk_ids, target_chunk_ids,
                    src_row["src_lang"], src_row["tgt_lang"],
                    min_src_words=min_src_words, min_src_chars_cjk=min_src_chars_cjk,
                )
                if not source_chunks or len(source_chunks) != len(target_chunks):
                    skipped += 1
                    continue

            # Fix 2 (2026-08-18): reassign latency based on our chunk count,
            # using EAST-inherited thresholds. Ensures the latency token is
            # self-consistent with cond-B's actual chunk density.
            inherited_latency = src_row["latency"]
            if reassign_latency:
                new_latency = latency_from_chunk_count(len(source_chunks))
                if new_latency != inherited_latency:
                    n_relabelled += 1
                    latency_flips[(inherited_latency, new_latency)] = \
                        latency_flips.get((inherited_latency, new_latency), 0) + 1
            else:
                new_latency = inherited_latency

            # Sanity: verify interleave doesn't raise.
            try:
                _ = interleave(EastRow(
                    source=src_row["source"], target=src_row["target"],
                    src_lang=src_row["src_lang"], tgt_lang=src_row["tgt_lang"],
                    latency=new_latency,
                    source_chunks=source_chunks, target_chunks=target_chunks,
                ))
            except Exception:
                skipped += 1
                continue

            kept.append({
                "index": idx,
                "source": src_clean,
                "target": tgt_clean,
                "src_lang": src_row["src_lang"],
                "tgt_lang": src_row["tgt_lang"],
                "latency": new_latency,
                "source_chunks": source_chunks,
                "target_chunks": target_chunks,
                # v4 fix (2026-08-19): raw BPE ids per chunk. Downstream
                # SFT builds input_ids directly from these to avoid the
                # decode+retokenize round-trip artifact (chunks ending in
                # `.` retokenize as `▁.` instead of `.`, misaligning
                # training with streaming inference).
                "source_chunk_ids": source_chunk_ids,
                "target_chunk_ids": target_chunk_ids,
                # Provenance so downstream analysis can inspect the fix effects.
                "_annotator_meta": {
                    "tau_used": tau_used,
                    "fell_back_from_primary_tau": fell_back,
                    "inherited_latency": inherited_latency,
                    "reassigned_latency": reassign_latency,
                },
            })

    return kept, {
        "missing": n_missing,
        "skipped": skipped,
        "collapse_at_primary_tau": n_collapse_at_primary,
        "still_collapse_after_fallback": n_still_collapse,
        "relabelled": n_relabelled,
        "latency_flips": latency_flips,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path, nargs="+",
                    default=[REPO_ROOT / "results" / "phase2" / "annot_ot_n2k" / "matrices.jsonl"],
                    help="One or more matrices.jsonl files. For multilingual v5, pass "
                         "all 10 direction files (or use shell-glob: --matrices "
                         "results/phase2/annot_ot_multi_*/matrices.jsonl).")
    ap.add_argument("--corpus_json", type=Path, default=None,
                    help="Override the default SiMT-De-En-660K source lookup with a "
                         "custom pool JSON (e.g. multilingual_source_pool_v5.json). "
                         "Indices in matrices.jsonl must resolve within this pool.")
    ap.add_argument("--tokenizer_path", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--tau", type=float, default=0.30,
                    help="Primary OT divergence threshold. Default 0.30 (Gate-1 Config F primary).")
    ap.add_argument("--tau_fallbacks", type=str, default="0.50,0.70,1.00",
                    help="Comma-separated tau values to fall back on when the "
                         "primary tau produces a single-chunk collapse. Default "
                         "0.50,0.70,1.00 (2026-08-18 collapse fix). Set to empty "
                         "string to disable fallback (pre-fix behavior).")
    ap.add_argument("--no_reassign_latency", action="store_true",
                    help="Do NOT reassign latency labels based on our chunk count. "
                         "Default: reassign per EAST-inherited chunk-count thresholds "
                         "(<=3 -> high, 4-5 -> medium, >=6 -> low) so the latency "
                         "token is self-consistent with cond-B chunks (2026-08-18 fix).")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "sft_dataset_n2k.json")
    ap.add_argument("--merge_small_chunks", action="store_true",
                    help="EAST §3.1 post-processing: merge any chunk with < min_src_words "
                         "source words (or < min_src_chars_cjk chars for CJK) into next chunk. "
                         "Brings OT chunks closer to GPT-4's semantic-unit size.")
    ap.add_argument("--min_src_words", type=int, default=2,
                    help="Chunks with FEWER than this many source words are merged. "
                         "Default 2 = EAST rule (merge 1-word slivers). Set to 4 "
                         "to merge chunks with <=3 words (aggressive).")
    ap.add_argument("--min_src_chars_cjk", type=int, default=4,
                    help="CJK-source variant of --min_src_words. Default 4 = EAST rule.")
    ap.add_argument("--augment_latency", action="store_true",
                    help="2026-08-19 v4 augmentation: for each row with >=4 "
                         "chunks, generate merged versions with fewer chunks "
                         "(higher latency labels). Same source content appears "
                         "at multiple latencies — teaches the model to condition "
                         "chunk granularity on the <|latency|> token.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    corpus_path = args.corpus_json if args.corpus_json is not None else CORPUS
    print(f"Loading source-lookup corpus from {corpus_path}", flush=True)
    with open(corpus_path) as f:
        corpus_rows = json.load(f)
    corpus_by_idx = {r["index"]: r for r in corpus_rows}
    print(f"  {len(corpus_by_idx):,} rows indexed", flush=True)

    # Assemble tau ladder: primary first, then any fallbacks.
    fallbacks = [float(x) for x in args.tau_fallbacks.split(",") if x.strip()]
    tau_ladder = [args.tau] + [t for t in fallbacks if t > args.tau]
    reassign_latency = not args.no_reassign_latency

    matrices_paths = args.matrices if isinstance(args.matrices, list) else [args.matrices]
    print(f"Building cond-B dataset from {len(matrices_paths)} matrices file(s):", flush=True)
    for mp in matrices_paths:
        print(f"  - {mp}", flush=True)
    print(f"  tau ladder: {tau_ladder} (primary={args.tau}, fallbacks tried in order)", flush=True)
    print(f"  reassign_latency: {reassign_latency} "
          f"(EAST-inherited thresholds: <=3 high, 4-5 medium, >=6 low)", flush=True)

    kept, stats = [], None
    from collections import Counter
    combined_stats = {
        "skipped": 0, "missing": 0, "collapse_at_primary_tau": 0,
        "still_collapse_after_fallback": 0, "relabelled": 0,
        "latency_flips": Counter(),
    }
    for mp in matrices_paths:
        kept_i, stats_i = build_dataset(mp, tau_ladder, tokenizer, corpus_by_idx,
                                        reassign_latency=reassign_latency,
                                        merge_small=args.merge_small_chunks,
                                        min_src_words=args.min_src_words,
                                        min_src_chars_cjk=args.min_src_chars_cjk)
        kept.extend(kept_i)
        for k in ["skipped", "missing", "collapse_at_primary_tau",
                  "still_collapse_after_fallback", "relabelled"]:
            combined_stats[k] += stats_i.get(k, 0)
        if stats_i.get("latency_flips"):
            for kk, vv in stats_i["latency_flips"].items():
                combined_stats["latency_flips"][kk] += vv
    stats = combined_stats
    n = len(kept)
    print(f"  kept={n}  skipped={stats['skipped']}  missing={stats['missing']}", flush=True)
    print(f"  collapse-at-primary-tau: {stats['collapse_at_primary_tau']}  "
          f"({stats['collapse_at_primary_tau']*100/max(n,1):.1f}% of kept rows fell back)", flush=True)
    print(f"  still-collapse-after-fallback (dropped): {stats['still_collapse_after_fallback']}", flush=True)
    print(f"  latency relabelled: {stats['relabelled']}/{n} "
          f"({stats['relabelled']*100/max(n,1):.1f}%)", flush=True)
    if stats["latency_flips"]:
        print(f"  latency flip table (old -> new: count):", flush=True)
        for (old, new), cnt in sorted(stats["latency_flips"].items(), key=lambda x: -x[1]):
            print(f"    {old:>6s} -> {new:>6s}: {cnt}", flush=True)

    # v4 augmentation: for each row with >=4 chunks, generate merged versions
    # at higher latency labels. Same source content seen at multiple latencies
    # teaches the model to actually condition chunk granularity on <|latency|>.
    if args.augment_latency:
        n_base = len(kept)
        aug_rows = []
        for r in kept:
            aug_rows.extend(augment_row_at_lower_chunk_counts(r))
        kept.extend(aug_rows)
        print(f"\nAugmentation: +{len(aug_rows)} merged-chunk rows added "
              f"(base {n_base} -> total {len(kept)}, +{100*len(aug_rows)/max(n_base,1):.0f}%)",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)", flush=True)

    # Diagnostic — chunk-count and per-latency distribution AFTER fixes+augmentation.
    from collections import Counter
    n_final = len(kept)
    cc_dist = Counter(len(r["source_chunks"]) for r in kept)
    print(f"\nChunk-count distribution (top 12):", flush=True)
    for c, cnt in sorted(cc_dist.items())[:12]:
        print(f"  {c:>3d} chunks: {cnt} rows", flush=True)
    lat_dist = Counter(r["latency"] for r in kept)
    print(f"\nAssigned latency distribution:", flush=True)
    for lat in ["low", "medium", "high"]:
        print(f"  {lat:>6s}: {lat_dist.get(lat,0):>5d}  ({lat_dist.get(lat,0)*100/max(n_final,1):.1f}%)", flush=True)
    # Fallback-tau distribution (base rows only — augmented rows share meta)
    tau_dist = Counter(r["_annotator_meta"].get("tau_used") for r in kept
                        if not r["_annotator_meta"].get("augmented_from_base"))
    print(f"\nTau-used distribution (base rows only):", flush=True)
    n_base = sum(tau_dist.values())
    for tau_v in sorted(tau_dist):
        print(f"  tau={tau_v}: {tau_dist[tau_v]:>5d} rows  ({tau_dist[tau_v]*100/max(n_base,1):.1f}%)", flush=True)
    # Augmentation split
    n_aug = sum(1 for r in kept if r["_annotator_meta"].get("augmented_from_base"))
    print(f"\nBase rows: {n_final - n_aug}  Augmented rows: {n_aug}", flush=True)


if __name__ == "__main__":
    main()
