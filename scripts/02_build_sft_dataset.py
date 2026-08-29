"""
Build the cond-B SFT training corpus from our OT annotator's matrices.

Reads:
  results/_archive/v6b_gemma_2b/annot_ot_n2k/matrices.jsonl
Writes:
  results/_archive/v6b_gemma_2b/sft_dataset_n2k.json  — same schema as SiMT-660K.json
                                            but with our-annotator chunks

The output can be fed to `src/train/sft.py --corpus_file <path>`.

Chunk derivation. For each sentence:
  1. Load the (n, m) divergence matrix.
  2. Commit at a chosen tau per METHOD §4 (commit_from_matrix + enforce_monotone).
  3. Group consecutive commit points into (source_chunk, target_chunk) pairs via
     _chunks_from_commit (same routine used by the annotator online).

τ strategy — start with the tightest fixed-τ policy that avoids collapse.
Gate-1 Config F used τ=0.30 as the primary; that's what we ship as default.

Latency label (2026-08-22 rebucketing). Assigned via `latency_from_chunk_stats(cc, sw)`
using the empirical (cc, sw) rule fit to condA:
    cc <= 2                             -> high
    else cc / src_words >= 0.20         -> low
    else cc / src_words >= 0.13         -> medium
    else                                -> high
Optional `--augment_latency` produces coarsened copies of many-chunk rows to
expose the same source at higher latency labels. Base + aug rows use the
SAME rule, so labels are consistent across the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _chunks_from_commit, _enforce_monotone, _is_cjk_lang
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT


# Target-side function-word lists for stranded-endings merge (2026-08-23).
# A chunk ending in one of these words is a mid-NP/PP cutoff — merge it into
# the next chunk to move the boundary to a syntactically meaningful position.
# Lowercased match on the last whitespace-token of the target chunk.
STRANDED_ENDINGS: dict[str, set[str]] = {
    "en": {
        "the", "a", "an",
        "and", "or", "but", "nor", "so", "yet",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "into", "onto", "upon", "over", "under", "about", "as", "than",
        "that", "which", "who", "whom", "whose", "if", "when", "while",
        "though", "although", "because", "unless", "until", "since",
    },
    "de": {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einen", "einem", "einer", "eines",
        "und", "oder", "aber", "sondern", "denn",
        "in", "an", "auf", "von", "zu", "mit", "für", "über", "unter",
        "bei", "aus", "nach", "vor", "durch", "gegen", "ohne", "um",
        "dass", "wenn", "als", "weil", "obwohl", "damit",
    },
    "ru": {
        "и", "а", "но", "или", "да", "ни",
        "в", "во", "на", "с", "со", "по", "для", "к", "ко", "о", "об",
        "от", "до", "у", "из", "за", "перед", "над", "под", "между",
        "что", "чтобы", "если", "когда", "хотя", "потому",
    },
    # Arabic and Vietnamese lists deferred — target-side function-word
    # conventions differ enough that we want native-speaker review first.
}


def _chunk_ends_with_stopword(chunk_str: str, tgt_lang: str) -> bool:
    stops = STRANDED_ENDINGS.get(tgt_lang)
    if not stops:
        return False
    s = chunk_str.rstrip().rstrip(".,;:!?")
    if not s:
        return False
    last = s.split()[-1].lower() if s.split() else ""
    # Strip trailing punctuation from the last token too (e.g., "and,")
    last = last.rstrip(".,;:!?)")
    return last in stops


def merge_stranded_function_word_chunks(source_chunks, target_chunks,
                                        source_chunk_ids, target_chunk_ids,
                                        src_lang: str, tgt_lang: str):
    """Merge any (source_chunk[i], target_chunk[i]) whose target chunk ends
    with a syntactic function word (article/preposition/conjunction) into
    the following pair — the boundary lands in a mid-NP/PP dead-zone
    otherwise.

    Iterates until no more merges apply. If the LAST chunk ends with a
    stopword, merge it into the previous chunk instead (sentence-final
    chunks should end cleanly).

    Operates in lock-step on source, target, and both id lists so byte-
    round-trip is preserved on both sides.
    """
    stops = STRANDED_ENDINGS.get(tgt_lang)
    if not stops:
        return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids

    src_sep = "" if _is_cjk_lang(src_lang) else " "
    tgt_sep = "" if _is_cjk_lang(tgt_lang) else " "

    # Forward pass: merge chunk i into chunk i+1 iteratively
    changed = True
    while changed and len(target_chunks) > 1:
        changed = False
        for i in range(len(target_chunks) - 1):
            if _chunk_ends_with_stopword(target_chunks[i], tgt_lang):
                target_chunks[i + 1] = (target_chunks[i] + tgt_sep + target_chunks[i + 1]).strip()
                source_chunks[i + 1] = (source_chunks[i] + src_sep + source_chunks[i + 1]).strip()
                target_chunk_ids[i + 1] = target_chunk_ids[i] + target_chunk_ids[i + 1]
                source_chunk_ids[i + 1] = source_chunk_ids[i] + source_chunk_ids[i + 1]
                del target_chunks[i]
                del source_chunks[i]
                del target_chunk_ids[i]
                del source_chunk_ids[i]
                changed = True
                break

    # Last-chunk cleanup: if the final chunk ends with a stopword, merge into previous
    if len(target_chunks) > 1 and _chunk_ends_with_stopword(target_chunks[-1], tgt_lang):
        target_chunks[-2] = (target_chunks[-2] + tgt_sep + target_chunks[-1]).strip()
        source_chunks[-2] = (source_chunks[-2] + src_sep + source_chunks[-1]).strip()
        target_chunk_ids[-2] = target_chunk_ids[-2] + target_chunk_ids[-1]
        source_chunk_ids[-2] = source_chunk_ids[-2] + source_chunk_ids[-1]
        del target_chunks[-1]
        del source_chunks[-1]
        del target_chunk_ids[-1]
        del source_chunk_ids[-1]

    return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids

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


# Latency rule (2026-08-22 rebuild): use the chunk-density signal that
# GPT-4 empirically uses in cond-A, not raw chunk count. Prior rule
# (<=3 high, 4-6 medium, >=7 low) baked in a source-length confound —
# long sentences ended up in `low` even when their granularity (source
# words per chunk) was medium, and short sentences monopolised `high`.
# See LOG.md 2026-08-22 "Latency rebucketing" for the P(lat|cc,sw)
# derivation.
#
# Rule:
#   cc <= LATENCY_CC1_MAX                     -> high   (few chunks -> long wait)
#   else cc / src_words >= LATENCY_LOW_CCSW   -> low    (aggressive commit)
#   else cc / src_words >= LATENCY_MED_CCSW   -> medium
#   else                                       -> high
#
# Thresholds fit on condA joint P(latency | cc, sw). 88% accuracy on
# condA, 60% on Multi-90K, marginals reproduce condA within 1pp.
LATENCY_CC1_MAX = 2      # cc <= 2 -> `high` regardless of length
LATENCY_LOW_CCSW = 0.20  # cc / src_words >= 0.20 -> `low`
LATENCY_MED_CCSW = 0.13  # cc / src_words in [0.13, 0.20) -> `medium`


def latency_from_chunk_stats(cc: int, src_words: int) -> str:
    """Assign a latency bucket from chunk count + source length.

    Uses the empirical GPT-4 rule extracted from condA (2026-08-22):
      - cc <= 2                             -> `high`
      - else cc / src_words >= 0.20         -> `low`
      - else cc / src_words >= 0.13         -> `medium`
      - else                                -> `high`
    """
    if cc <= LATENCY_CC1_MAX:
        return "high"
    ratio = cc / max(src_words, 1)
    if ratio >= LATENCY_LOW_CCSW:
        return "low"
    if ratio >= LATENCY_MED_CCSW:
        return "medium"
    return "high"


def _count_source_words(source: str, src_lang: str) -> int:
    """Source-word count for the latency rule. Matches condA (space-split
    for alphabetic scripts, character-count for CJK)."""
    if _is_cjk_lang(src_lang):
        return sum(1 for c in source if not c.isspace())
    return len(source.split())


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

    Since coarsening reduces cc while sw is fixed, cc/sw decreases, so the
    label naturally walks up the ladder: `low` -> `medium` -> `high`. Each
    aug row is labelled using the same (cc, sw) rule as the base builder
    (`latency_from_chunk_stats`), so downstream training sees consistent
    (label, chunk-density) semantics across base and augmentation rows.

    Merge targets:
      - k >= 4  -> merge to ceil(k/2) chunks    (one coarser step)
      - k >= 7  -> also merge to ceil(k/4) chunks (two coarser steps)
    We also try target_n = LATENCY_CC1_MAX (=2) explicitly so any row with
    k > 2 can produce a `high`-labelled augmentation, regardless of divisor.
    Duplicate target_n values are de-duplicated.

    New rows inherit index (with `_aug` suffix in meta), source/target
    strings, but have merged chunks and a freshly-assigned latency label.
    Rows whose label doesn't change are skipped (no informational gain).
    """
    import copy
    k = len(row["source_chunks"])
    if k <= LATENCY_CC1_MAX:
        return []  # already `high` under the (cc, sw) rule; no coarsening possible

    sw = _count_source_words(row["source"], row.get("src_lang", "en"))
    base_latency = row["latency"]

    # Candidate target_n values, ordered by aggressiveness.
    candidates = []
    if k >= 4:
        candidates.append(("aug2", -(-k // 2)))          # halve
    if k >= 7:
        candidates.append(("aug4", -(-k // 4)))          # quarter
    candidates.append(("aug_cc1", LATENCY_CC1_MAX))       # force to `high` bucket
    # de-duplicate by target_n, preserve first tag seen
    seen = set()
    dedup = []
    for tag, tn in candidates:
        if tn < 1 or tn >= k or tn in seen:
            continue
        seen.add(tn)
        dedup.append((tag, tn))

    out = []
    for tag, target_n in dedup:
        new_src, new_tgt, new_src_ids, new_tgt_ids = merge_chunks_to_n(
            row["source_chunks"], row["target_chunks"],
            row["source_chunk_ids"], row["target_chunk_ids"],
            target_n,
            src_lang=row.get("src_lang", "en"),
            tgt_lang=row.get("tgt_lang", "en"),
        )
        if not new_src:
            continue
        new_latency = latency_from_chunk_stats(len(new_src), sw)
        if new_latency == base_latency:
            continue  # no informational gain — same label as base
        # Skip if we already emitted an aug with the same target label
        # (aug2 and aug4 may both land in `medium` for some k).
        if any(o["latency"] == new_latency for o in out):
            continue
        aug = copy.copy(row)
        aug["source_chunks"] = new_src
        aug["target_chunks"] = new_tgt
        aug["source_chunk_ids"] = new_src_ids
        aug["target_chunk_ids"] = new_tgt_ids
        aug["latency"] = new_latency
        aug_meta = dict(row.get("_annotator_meta", {}))
        aug_meta["augmented_from_base"] = True
        aug_meta["base_n_chunks"] = k
        aug_meta["base_src_words"] = sw
        aug_meta["merged_to_n_chunks"] = len(new_src)
        aug_meta["merge_tag"] = tag
        aug_meta["base_latency"] = base_latency
        aug["_annotator_meta"] = aug_meta
        out.append(aug)
    return out


def build_dataset(matrices_path: Path, tau_ladder: list[float], tokenizer,
                  corpus_by_idx: dict, reassign_latency: bool = True,
                  merge_small: bool = False, min_src_words: int = 2,
                  min_src_chars_cjk: int = 4, merge_stranded: bool = False,
                  refine_bounds: bool = False, refine_window: int = 3,
                  refine_alpha: float = 1.0, refine_beta: float = 1.0,
                  keep_collapsed: bool = False,
                  force_latency: Optional[str] = None):
    """Return list of dicts matching SiMT-660K.json schema.

    `tau_ladder`: list of tau values, tried in order per row. First tau that
    produces > 1 chunks is used (collapse fallback, 2026-08-18 fix). Primary
    tau is tau_ladder[0]; the rest are fallbacks.
    `reassign_latency`: if True (default), overwrite each row's latency
    label using `latency_from_chunk_stats(cc, sw)` — the (cc, sw) rule
    empirically fit to condA (GPT-4 chunks). If False, inherit SiMT-660K's
    original label (which was GPT-4-derived for the underlying source but
    may be inconsistent with our OT chunks). Pre-2026-08-22 defaults used
    a pure chunk-count rule; that produced a source-length confound
    (see LOG.md 2026-08-22 rebucketing entry).
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
                n_still_collapse += 1
                if not keep_collapsed:
                    # Even the largest tau in the ladder collapsed. Drop.
                    skipped += 1
                    continue
                # else: keep this cc=1 row — it's a legitimate "read the whole
                # source then emit" high-latency training example. Useful for
                # tau-sweep balancing where cc=1 rows are the natural high-
                # latency variant of a source.

            src_clean = src_row["source"].strip()
            tgt_clean = src_row["target"].strip()

            # 2026-08-22 v6b fix: use the annotator's ORIGINAL tokenization
            # (no leading space — how it fed the divergence matrix). The
            # v4/v5 "prepend leading space and re-tokenize" gate was a
            # streaming-alignment kludge for v1-v5 (which fed raw source
            # after `<|latency|>` in the prompt — leading space imputed by
            # the string join). v6 goes to direct-ids splice at both
            # training and inference: `src/train/sft.py` builds
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

            # 2026-08-23 OT-guided boundary refinement (opt-in).
            # Shifts each chunk boundary within a ±window by voting on
            # (OT confidence × syntactic goodness). Uses the raw OT matrix
            # `rec["matrix"]` — must run BEFORE _chunks_from_commit while
            # the matrix + commit are still aligned. Preserves monotonicity.
            # Disabled by default; removable by omitting the CLI flag.
            if refine_bounds:
                from src.annotator.boundary_refine import refine_boundaries
                commit = refine_boundaries(
                    commit, src_ids_orig, rec["matrix"], tokenizer,
                    tau=tau_used,
                    window=refine_window, alpha=refine_alpha, beta=refine_beta,
                    stranded_endings=None,   # post-emit handles target-side
                    src_lang=src_row["src_lang"],
                )

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

            # 2026-08-23: stranded-function-word merge. If a target chunk ends
            # with a determiner/preposition/conjunction (mid-NP/PP dead zone),
            # merge it into the next chunk. Only fires for target langs with a
            # curated stopword list (en/de/ru today).
            #
            # When --refine_boundaries is active, this pass is subsumed as the
            # `syn(i) < 0` branch. Kept as an independent flag so callers can
            # enable stranded-merge WITHOUT boundary refinement (matches the
            # rb_fw ablation cell), or vice versa.
            if merge_stranded and len(source_chunks) > 1:
                source_chunks, target_chunks, source_chunk_ids, target_chunk_ids = merge_stranded_function_word_chunks(
                    source_chunks, target_chunks, source_chunk_ids, target_chunk_ids,
                    src_row["src_lang"], src_row["tgt_lang"],
                )
                if not source_chunks or len(source_chunks) != len(target_chunks):
                    skipped += 1
                    continue

            # Reassign latency using the empirical (cc, sw) rule fit to condA
            # (2026-08-22 rebucketing). Prior versions used chunk-count-only
            # thresholds, which correlated `low` with long sentences rather
            # than with commit granularity.
            inherited_latency = src_row["latency"]
            if force_latency is not None:
                # tau-sweep balanced mode: override with the caller-supplied label
                new_latency = force_latency
                if new_latency != inherited_latency:
                    n_relabelled += 1
                    latency_flips[(inherited_latency, new_latency)] = \
                        latency_flips.get((inherited_latency, new_latency), 0) + 1
            elif reassign_latency:
                sw = _count_source_words(src_clean, src_row["src_lang"])
                new_latency = latency_from_chunk_stats(len(source_chunks), sw)
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
                         "results/_archive/v6b_gemma_2b/annot_ot_multi_*/matrices.jsonl).")
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
                    help="Do NOT reassign latency labels. Default: reassign using "
                         "the empirical (cc, src_words) rule fit to condA — "
                         "cc<=2 -> high; else cc/sw>=0.20 low, >=0.13 medium, else high. "
                         "This is the 2026-08-22 rebucketing (see LOG.md). Set to skip "
                         "reassignment and inherit SiMT-660K's original label.")
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
    ap.add_argument("--force_latency", type=str, default=None,
                    choices=[None, "low", "low-medium", "medium", "medium-high", "high"],
                    help="Override the latency label for EVERY row to this value. "
                         "Use with tau-sweep balanced augmentation: run one build "
                         "per tau value with --force_latency <corresponding-bucket>, "
                         "then concatenate. Bypasses --no_reassign_latency and any "
                         "(cc, sw) rule; the tau is the label.")
    ap.add_argument("--keep_collapsed", action="store_true",
                    help="Keep rows where OT collapses to a single chunk (cc=1). "
                         "Default: drop such rows (they're rare edge cases). "
                         "For tau-sweep balanced augmentation, cc=1 rows are the "
                         "natural high-latency variant of each source and should "
                         "be kept — enable this flag.")
    ap.add_argument("--refine_boundaries", action="store_true",
                    help="2026-08-23: OT-guided boundary voting. For each OT-"
                         "chosen chunk boundary, search within ±window source "
                         "tokens and pick the position maximising "
                         "α · ot_confidence + β · syntactic_score. Uses raw "
                         "OT divergence values (still purely backbone-derived) "
                         "combined with a language-agnostic syntactic signal "
                         "(post-punct=+1, post-comma=+0.5). Non-destructive: "
                         "omit the flag to disable and reproduce prior behavior. "
                         "See src/annotator/boundary_refine.py.")
    ap.add_argument("--refine_window", type=int, default=3,
                    help="Search window (source tokens) around each OT commit "
                         "position during --refine_boundaries. Default 3.")
    ap.add_argument("--refine_alpha", type=float, default=1.0,
                    help="Weight for OT-confidence in the voting score.")
    ap.add_argument("--refine_beta", type=float, default=1.0,
                    help="Weight for syntactic-goodness in the voting score.")
    ap.add_argument("--merge_stranded_function_words", action="store_true",
                    help="2026-08-23 fix. Merge any (source,target) chunk pair "
                         "whose TARGET chunk ends with a stranded determiner/"
                         "preposition/conjunction (e.g. 'the', 'and', 'of', 'in') "
                         "into the next chunk. Moves the boundary from a mid-NP "
                         "dead-zone to a syntactically meaningful position. "
                         "Cond-A has 1.6%% stranded-endings; ours has 20.2%% "
                         "without this fix (see LOG.md 2026-08-23). Only fires "
                         "for target languages with a curated stopword list "
                         "(en/de/ru today; ar/vi deferred pending native review).")
    ap.add_argument("--augment_latency", action="store_true",
                    help="Coarsen chunks to expose the SAME source at multiple "
                         "latency labels. Under the (cc, sw) rule (2026-08-22), "
                         "coarsening lowers cc/sw and walks the label up: low -> "
                         "medium -> high. For each row with cc > 2 we try up to "
                         "3 coarser variants (ceil(k/2), ceil(k/4), and 2 chunks) "
                         "and keep any that produce a NEW latency label. Directly "
                         "teaches the model to condition streaming granularity on "
                         "the natural-language latency prompt — analogous to condA's "
                         "P(latency|cc,sw) overlap.")
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
                                        min_src_chars_cjk=args.min_src_chars_cjk,
                                        merge_stranded=args.merge_stranded_function_words,
                                        refine_bounds=args.refine_boundaries,
                                        refine_window=args.refine_window,
                                        refine_alpha=args.refine_alpha,
                                        refine_beta=args.refine_beta,
                                        keep_collapsed=args.keep_collapsed,
                                        force_latency=args.force_latency)
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
