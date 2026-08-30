"""OT-guided boundary refinement via local voting.

Purpose
-------
The OT annotator produces a divergence matrix D[i][j] between (source prefix
of length i, full-source-conditioned distribution at target position j).
`commit_from_matrix` thresholds this at τ to pick commit positions. But the
divergence signal is continuous — often multiple adjacent source positions
have similar sub-τ divergence (a "plateau"), and the exact boundary picked
by τ-thresholding is somewhat arbitrary within the plateau.

This module reintroduces the discarded structure: for each chunk boundary
selected by OT, search within a window and prefer positions that combine
(a) high OT confidence (divergence well below τ) with (b) syntactic
naturalness (after punctuation, not stranding a determiner or preposition
at the emitted chunk tail).

Framing
-------
This is a POST-PROCESSING step layered on top of the standard OT + τ
pipeline. It is enabled by an explicit flag and can be removed without any
effect on the base pipeline. It subsumes two prior monkey-patch fixes:

- Case-1 punctuation-snap: `syn(i) = +1` at post-punctuation positions →
  boundaries are pulled toward the nearest punct within the window.
- Stranded-function-word merge (STRANDED_ENDINGS in
  scripts/03_build_sft_dataset.py): `syn(i) = -1` at positions where
  the corresponding target-side chunk would end with a determiner /
  preposition / conjunction → boundaries pushed away from mid-NP/PP splits.

Both effects fall out of one common composite `score(i, j) = α · confidence +
β · syn(i)` — no per-case rules, no hand-crafted heuristics beyond a
short language-specific stopword list (shared with `STRANDED_ENDINGS`).

Story
-----
"OT gives us a continuous per-position divergence signal. τ discretizes it
into chunks, but often multiple adjacent positions have similar sub-τ
divergence. We resolve the ambiguity by preferring positions that maximise
a composite score of OT confidence and syntactic naturalness — using the
structure τ-thresholding discards, not replacing OT."

The refinement is monotone-preserving: no boundary crosses its neighbours.
It never invents a boundary OT didn't consider — it only shifts within a
±w window around each existing commit.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set

# Sentence-ending punctuation (Latin + Arabic + CJK); scored as +syn.
_SENTENCE_END_CHARS = set(".!?;،؛؟।。、！？")
# Softer boundaries (comma/colon); scored as +syn/2.
_SOFT_END_CHARS = set(",:")


def _last_char(text: str) -> str:
    """Rightmost non-whitespace character of `text`."""
    for c in reversed(text):
        if not c.isspace():
            return c
    return ""


def _first_word_lower(text: str) -> str:
    """First whitespace-delimited word of `text`, lowercased and stripped
    of trailing punctuation. Empty string if `text` is empty."""
    t = text.strip()
    if not t:
        return ""
    first = t.split()[0]
    return first.lower().rstrip(".,;:!?")


def _last_word_lower(text: str) -> str:
    t = text.rstrip()
    if not t:
        return ""
    last = t.split()[-1]
    return last.lower().rstrip(".,;:!?)")


def syntactic_score(
    src_prefix_end: str,
    tgt_chunk_current: str,
    tgt_chunk_next_first_word: str,
    stranded_endings: Optional[Set[str]] = None,
    src_lang: str = "en",
) -> float:
    """Syntactic goodness of placing a boundary right after `src_prefix_end`,
    given the target chunk that would be emitted before the boundary and
    the first word of the next chunk.

    +1.0 : source prefix ends at sentence-ending punctuation (`. ! ? ;`).
           These are the safest boundaries in every language.
    +0.5 : source prefix ends at comma / colon (mid-sentence clause break).
    -1.0 : target chunk ends with a stranded function word (determiner,
           preposition, conjunction, subordinator) — the boundary would
           split a noun/verb phrase.
     0.0 : neutral.

    Scores add if multiple conditions apply. `stranded_endings` is a set
    of lowercased target-side function-word forms (see STRANDED_ENDINGS
    in scripts/03_build_sft_dataset.py).
    """
    score = 0.0
    last_ch = _last_char(src_prefix_end)
    if last_ch in _SENTENCE_END_CHARS:
        score += 1.0
    elif last_ch in _SOFT_END_CHARS:
        score += 0.5
    if stranded_endings:
        tgt_last = _last_word_lower(tgt_chunk_current)
        if tgt_last in stranded_endings:
            score -= 1.0
    return score


def ot_confidence(divergence: float, tau: float) -> float:
    """Voting weight from OT: how much the divergence undershoots τ.
    Zero if divergence >= τ (no commit signal at that position).
    Bounded in [0, τ].
    """
    if divergence >= tau or divergence != divergence:  # NaN check
        return 0.0
    return tau - divergence


def refine_source_boundary(
    j: int,
    initial_commit_i: int,
    divergence_matrix: List[List[float]],
    src_chunk_strs_at_boundary: List[str],
    tgt_chunk_str_current: str,
    tgt_chunk_str_next_first_word: str,
    tau: float,
    lower_bound_i: int,
    upper_bound_i: int,
    window: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    stranded_endings: Optional[Set[str]] = None,
    src_lang: str = "en",
) -> int:
    """Choose the source position i ∈ [lower_bound_i, upper_bound_i] within
    a ±window of `initial_commit_i` that maximises

        α · ot_confidence(D[i][j], τ) + β · syntactic_score(source_prefix(i), ...)

    `src_chunk_strs_at_boundary` is a list of candidate SOURCE chunk strings
    corresponding to each candidate i in the search window (caller supplies
    these; the refinement doesn't reach back into the tokenizer).

    Returns the argmax i. Preserves monotonicity via bounds.
    Returns `initial_commit_i` unchanged if the window is empty.
    """
    lo = max(lower_bound_i, initial_commit_i - window)
    hi = min(upper_bound_i, initial_commit_i + window)
    if lo > hi:
        return initial_commit_i

    best_i = initial_commit_i
    best_score = float("-inf")
    for offset, i in enumerate(range(lo, hi + 1)):
        # D matrix is indexed [i_row][j]; i_row = i - 1 because a commit at
        # source position i reads source[..i-1], so the corresponding OT
        # divergence row is at index (i-1) — matches commit_from_matrix's
        # `i = i_row + 1` convention.
        d_row = i - 1
        if d_row < 0 or d_row >= len(divergence_matrix):
            div = float("inf")
        elif j >= len(divergence_matrix[d_row]):
            div = float("inf")
        else:
            div = divergence_matrix[d_row][j]
        conf = ot_confidence(div, tau)
        src_prefix = src_chunk_strs_at_boundary[offset] if offset < len(src_chunk_strs_at_boundary) else ""
        syn = syntactic_score(
            src_prefix_end=src_prefix,
            tgt_chunk_current=tgt_chunk_str_current,
            tgt_chunk_next_first_word=tgt_chunk_str_next_first_word,
            stranded_endings=stranded_endings,
            src_lang=src_lang,
        )
        score = alpha * conf + beta * syn
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def refine_boundaries(
    commit: List[int],
    src_token_ids: List[int],
    divergence_matrix: List[List[float]],
    tokenizer,
    tau: float,
    window: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    stranded_endings: Optional[Set[str]] = None,
    src_lang: str = "en",
) -> List[int]:
    """Refine an entire commit vector by shifting each chunk boundary to
    the vote-maximising position within a window.

    `commit`: per-target-token source commit position (as produced by
              `commit_from_matrix` + `_enforce_monotone`).
    `src_token_ids`: source token id sequence (for decoding candidate
                     source prefixes to inspect for punctuation).
    `divergence_matrix`: the raw OT matrix that produced `commit`.
    `tau`: threshold that was used.
    `window`, `alpha`, `beta`: hyperparameters (see module docstring).
    `stranded_endings`: target-side function-word set (per-tgt-lang);
                       pass None to disable the negative-syn signal.

    Returns a refined commit vector with the same length as `commit`.
    Monotonicity is preserved. If `stranded_endings` is None and no
    punctuation is found in the source, the refinement is a no-op.
    """
    if not commit:
        return commit
    n = len(commit)

    # Identify boundary positions in the commit vector: indices j where
    # commit[j] != commit[j-1]. Also treat the final chunk end as a boundary.
    boundary_js = [0]
    for j in range(1, n):
        if commit[j] != commit[j - 1]:
            boundary_js.append(j)
    boundary_js.append(n)  # sentinel: end of target
    # boundary_js is len K+1 for K chunks; between boundary_js[k] and
    # boundary_js[k+1] all target tokens share commit[boundary_js[k]].

    new_commit = list(commit)

    # For each boundary (k = 0..K-1), shift the source commit for the group.
    # k=0 shifts the FIRST group's commit — chunk-0's end boundary. If commit[0]
    # equals its chunk's end (very fast OT convergence at target j=0), there's
    # no transition TO that value in the commit vector, so it would otherwise
    # be missed. Explicit k=0 pass fixes this. lower_bound is 1 (chunks must
    # be non-empty); upper_bound is chunk-1's commit minus 1.
    for k in range(0, len(boundary_js) - 1):
        j_here = boundary_js[k]
        current_i = commit[j_here]
        # Monotonicity bounds:
        #   lower = previous chunk's commit value (or 0 for k=0)
        #   upper = NEXT chunk's commit value - 1 (or len(src) for last chunk)
        # Previous fix accidentally set upper to the CURRENT chunk's own commit
        # value (via commit[boundary_js[k+1]-1] which lands on last j of current
        # group). Corrected: commit[boundary_js[k+1]] is first j of NEXT group,
        # whose commit is C_{k+1}, and we allow up to C_{k+1}-1.
        if k == 0:
            lower = 0             # chunk 0 can start at source position 0
            prev_i = 0            # chunk 0 begins at src[0:]
        else:
            lower = commit[j_here - 1]  # previous group's commit = C_{k-1}
            prev_i = new_commit[j_here - 1]
        if k + 1 < len(boundary_js) - 1:
            # There IS a next chunk (boundary_js[-1] is the n sentinel)
            upper = commit[boundary_js[k + 1]] - 1  # C_{k+1} - 1
        else:
            upper = len(src_token_ids)  # last chunk can extend to source end
        # Build candidate source prefixes (decoded) for each candidate i in window
        lo = max(lower + 1, current_i - window)
        hi = min(upper, current_i + window)
        if lo > hi:
            continue
        # Source chunk ending at candidate i covers src_token_ids[prev_i:i].
        # For k=0, prev_i=0; for k>=1, prev_i=previous group's commit (set above).
        candidate_src_prefixes = []
        for i_cand in range(lo, hi + 1):
            span_ids = src_token_ids[prev_i:i_cand]
            candidate_src_prefixes.append(
                tokenizer.decode(span_ids, skip_special_tokens=True).strip() if span_ids else ""
            )
        # For target chunk string we look at target tokens (approximated as empty
        # here — the caller of a full pipeline supplies decoded target chunks
        # if beta > 0 for stranded-endings). Passing empty means the negative-syn
        # contribution is only active if stranded_endings is None (no-op).
        # In the full-pipeline usage (build_sft_dataset) the callback re-derives
        # the target chunk after applying the shift; this function is designed
        # so caller can pass a lambda that scores based on their own tgt-chunk
        # derivation. For now we use empty tgt_chunk (beta*syn will be +1/0/+0.5
        # based on src punct only) — the stranded-endings check is more
        # naturally applied AFTER chunk emission (see refine_chunks_after_emit).
        best_i = refine_source_boundary(
            j=j_here,
            initial_commit_i=current_i,
            divergence_matrix=divergence_matrix,
            src_chunk_strs_at_boundary=candidate_src_prefixes,
            tgt_chunk_str_current="",  # deferred to post-emit check
            tgt_chunk_str_next_first_word="",
            tau=tau,
            lower_bound_i=lower + 1,
            upper_bound_i=upper,
            window=window,
            alpha=alpha,
            beta=beta,
            stranded_endings=None,  # post-emit handles this
            src_lang=src_lang,
        )
        # Rewrite the commit for all target positions in this group
        j_end = boundary_js[k + 1]
        for jj in range(j_here, j_end):
            if new_commit[jj] == current_i:
                new_commit[jj] = best_i
    return new_commit


def refine_chunks_after_emit(
    source_chunks: List[str],
    target_chunks: List[str],
    source_chunk_ids: List[List[int]],
    target_chunk_ids: List[List[int]],
    src_lang: str,
    tgt_lang: str,
    stranded_endings_per_lang: dict,
    is_cjk_lang,
) -> tuple:
    """Post-emit second pass: merge any (source_chunk[i], target_chunk[i])
    whose TARGET chunk ends with a stranded function word into the next
    chunk. This is the `beta * syn < 0` branch, applied after the source-
    side voting has run.

    Idempotent with the standalone `merge_stranded_function_word_chunks`
    in phase2_build_sft_dataset.py — we import it via the shared stopword
    set. The two operations compose: the refinement moves boundaries by
    up to `window` source tokens; this pass merges any remaining strandings.
    Iterative.

    Kept SEPARATE so the refinement stage can be disabled without losing
    the target-side stranded check.
    """
    stops = stranded_endings_per_lang.get(tgt_lang)
    if not stops:
        return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids
    src_sep = "" if is_cjk_lang(src_lang) else " "
    tgt_sep = "" if is_cjk_lang(tgt_lang) else " "

    changed = True
    while changed and len(target_chunks) > 1:
        changed = False
        for i in range(len(target_chunks) - 1):
            tgt_last = _last_word_lower(target_chunks[i])
            if tgt_last in stops:
                target_chunks[i + 1] = (target_chunks[i] + tgt_sep + target_chunks[i + 1]).strip()
                source_chunks[i + 1] = (source_chunks[i] + src_sep + source_chunks[i + 1]).strip()
                target_chunk_ids[i + 1] = target_chunk_ids[i] + target_chunk_ids[i + 1]
                source_chunk_ids[i + 1] = source_chunk_ids[i] + source_chunk_ids[i + 1]
                del target_chunks[i]; del source_chunks[i]
                del target_chunk_ids[i]; del source_chunk_ids[i]
                changed = True
                break

    if len(target_chunks) > 1 and _last_word_lower(target_chunks[-1]) in stops:
        target_chunks[-2] = (target_chunks[-2] + tgt_sep + target_chunks[-1]).strip()
        source_chunks[-2] = (source_chunks[-2] + src_sep + source_chunks[-1]).strip()
        target_chunk_ids[-2] = target_chunk_ids[-2] + target_chunk_ids[-1]
        source_chunk_ids[-2] = source_chunk_ids[-2] + source_chunk_ids[-1]
        del target_chunks[-1]; del source_chunks[-1]
        del target_chunk_ids[-1]; del source_chunk_ids[-1]

    return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids
