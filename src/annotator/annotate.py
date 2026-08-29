"""
Teacher-free read/write annotator (METHOD §§1–4 + §6).

Given a parallel pair (source S, target T) and a backbone LLM M, decide
per-target-token commit points i*[j] by measuring when the predictive
distribution P_pre[i][j] = p_M(y_j | S_<=i, T_<j) has converged to the
full-source distribution P_full[j] = p_M(y_j | S, T_<j).

Contract with the rest of the codebase:
  * `annotate_pair()` returns a fully-populated `AnnotatedPair` including
    the commit trace, derived string chunks, and the interleaved EAST
    training string ready for SFT.
  * The annotator is the same LLM that will be fine-tuned (METHOD §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.annotator.criterion import CRITERIA, make_ot
from src.annotator.east_format import EastRow, interleave


# Separator inserted between source and target during the annotator's
# teacher-forced passes under the raw-concat prompt. Kept minimal — the
# exact template does not affect the commit criterion so long as it is
# identical in P_full and every P_pre[i] call (only the source portion
# changes length).
SEP = "\n"


def make_prompt_raw(src_prefix_str: str, src_lang: str, tgt_lang: str) -> str:
    """Raw-concat prompt: source_prefix + newline. No instruction. Used
    by the Phase-1 initial smoke; kept for A/B against chat-template."""
    return src_prefix_str + SEP


def make_prompt_chat(tokenizer, src_prefix_str: str, src_lang: str, tgt_lang: str) -> str:
    """Instruction-tuned prompt via the tokenizer's chat template.
    Applies the model's own chat format with a translation instruction,
    then appends the assistant generation prompt so the target continues
    from the assistant turn."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Translate the following {src_lang} text to {tgt_lang}.\n\n"
                f"{src_lang}: {src_prefix_str}\n\n"
                f"{tgt_lang}:"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@dataclass
class AnnotatedPair:
    source: str
    target: str
    src_lang: str
    tgt_lang: str
    latency: str

    # Per-target-token commit trace after monotonicity enforcement.
    commit_source_tok_idx: List[int] = field(default_factory=list)  # len = m
    # Divergence values D(P_full[j], P_pre[i][j]) at the fired i for each j.
    fired_divergence: List[float] = field(default_factory=list)  # len = m

    # Chunk strings derived from commit points, ready for interleave().
    source_chunks: List[str] = field(default_factory=list)
    target_chunks: List[str] = field(default_factory=list)

    east_str: str = ""

    # Diagnostics for METHOD §8 sanity checks.
    n_src_tok: int = 0
    n_tgt_tok: int = 0

    # Optional: full (n, m) divergence matrix D(P_full[j], P_pre[i][j]) —
    # populated when annotate_pair is called with return_full_matrix=True.
    # Enables offline tau sweeps without re-running forward passes.
    divergence_matrix: Optional[List[List[float]]] = None

    # Optional: (n, m) entropy matrix H(P_pre[i][j]) in nats — the
    # entropy-only ablation criterion (METHOD §3). Populated when
    # annotate_pair is called with record_entropy=True.
    entropy_matrix: Optional[List[List[float]]] = None
    # Full-source-conditioned target-token entropy H(P_full[j]) — one
    # per target token. Sanity anchor for the entropy sweep.
    entropy_full: Optional[List[float]] = None

    def as_east_row(self) -> EastRow:
        return EastRow(
            source=self.source,
            target=self.target,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            latency=self.latency,
            source_chunks=self.source_chunks,
            target_chunks=self.target_chunks,
        )


def _prob_at_positions(
    model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """One forward pass; return softmax probs at the requested positions.
    Shapes: input_ids (1, L); positions (m,); returns (m, V) on the model dtype."""
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0]  # (L, V)
    # softmax in float32 for numerical hygiene; keep on device.
    probs = F.softmax(logits[positions].float(), dim=-1)
    return probs


# 2026-08-22: KV-cache reuse across source-prefix lengths was investigated
# (see scripts/probe_annotator_kv_cache.py) and rejected for Gemma-4-family
# models. Two blockers:
#   1. Correctness: HybridCache (sliding-window + global attention + shared
#      KV) doesn't produce byte-identical logits under progressive extension
#      + snapshot-and-branch vs single full-forward. Divergence ~3-4% in
#      target probability distributions on de-en samples.
#   2. Performance: even without correctness issues, per-iteration cache
#      cloning (deepcopy of HybridCache pre-allocated buffers) dominates.
#      Measured 0.49× speedup (i.e., 2× slower) on Gemma-4-E2B.
# The naive per-prefix full-forward (below) is well-served by Gemma-4's
# fused attention kernels. Real speedups for annotation come from:
#   - Batching multiple sentences per forward pass (across sentences, not
#     across prefixes) — a separate optimization.
#   - Sharding annotation across multiple GPUs (already done via PBS).


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy H(P) in nats. Shape (m,) from (m, V)."""
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _enforce_monotone(commit: List[int]) -> List[int]:
    """i*[j] = max(i*[j], i*[j-1])."""
    out = []
    running = 0
    for c in commit:
        c = max(c, running)
        out.append(c)
        running = c
    return out


# Languages that don't use whitespace between words — SentencePiece's `▁`
# marker rarely appears, so every token position is effectively a valid
# streaming boundary (character-level or morpheme-level BPE).
CJK_LANGS = {
    "zh", "ja", "ko", "th", "km",
    "Chinese", "Japanese", "Korean", "Thai", "Khmer",
    "chinese", "japanese", "korean", "thai", "khmer",
}


def _is_cjk_lang(src_lang: str) -> bool:
    """True for scripts without whitespace word separators."""
    return src_lang in CJK_LANGS


def _is_word_boundary_before(src_token_ids: List[int], pos: int, tokenizer,
                              src_lang: str = "en") -> bool:
    """A commit at token position `pos` reads src_token_ids[..pos-1] and stops
    BEFORE src_token_ids[pos]. That's a valid word boundary iff pos is at the
    start of a whitespace-word — i.e., either pos == 0, pos == len (past end),
    or the token AT pos begins a new word.

    SentencePiece convention: a token begins a new word iff its piece starts
    with `▁` (U+2581). Punctuation-only tokens (no `▁` prefix) are treated as
    word-internal — but a "commit right after a period" position is really at
    (pos of period) + 1, i.e., between the period and the next token; if the
    next token starts with `▁` that IS a natural boundary.

    For CJK-family languages (zh/ja/ko/th/km) there is no whitespace and no
    `▁` marker, so **every token position is a valid streaming boundary**.
    """
    if pos <= 0 or pos >= len(src_token_ids):
        return True
    if _is_cjk_lang(src_lang):
        return True
    piece = tokenizer.convert_ids_to_tokens(src_token_ids[pos])
    return piece.startswith("▁")


def _snap_to_word_boundary(commit_i: int, src_token_ids: List[int], tokenizer,
                            prefer_after_punct: bool = True,
                            src_lang: str = "en") -> int:
    """Snap a raw commit position to the nearest valid word boundary AT OR
    AFTER commit_i (never backwards — the annotator's convergence signal
    fired here, so we should read at least this much).

    If `prefer_after_punct=True`, also look one position back: if the token
    IMMEDIATELY before commit_i is a punctuation-only token, the "natural"
    commit is right after the punctuation — which is what a raw commit_i
    that lands on a `▁`-prefixed token already is. So this is a no-op unless
    the raw commit is mid-word, in which case we scan forward for the next
    `▁` or end.

    For CJK-family languages this is a no-op (every position is a boundary).
    """
    if commit_i <= 0 or commit_i >= len(src_token_ids):
        return commit_i
    if _is_cjk_lang(src_lang):
        return commit_i
    # Already a word boundary?
    if _is_word_boundary_before(src_token_ids, commit_i, tokenizer, src_lang):
        return commit_i
    # Scan forward for the first `▁`-prefixed token or end.
    p = commit_i + 1
    while p < len(src_token_ids):
        if _is_word_boundary_before(src_token_ids, p, tokenizer, src_lang):
            return p
        p += 1
    return len(src_token_ids)


def _chunks_from_commit(
    commit: List[int],
    src_token_ids: List[int],
    tgt_token_ids: List[int],
    tokenizer,
    n_src_tok: int,
    snap_to_word_boundary: bool = True,
    src_lang: str = "en",
) -> tuple[list[str], list[str], list[list[int]], list[list[int]]]:
    """Group consecutive target tokens sharing a commit point into write
    chunks, and pair each with the read span that just became available.

    Returns 4-tuple: `(source_chunks, target_chunks, source_chunk_ids,
    target_chunk_ids)`. The `_ids` variants are the raw BPE token id lists
    for each chunk — downstream training pipelines that consume ids avoid
    the decode+retokenize round-trip artifact (2026-08-19 fix: chunks
    ending in `.` retokenize as `▁.` (id 783) after decode+strip vs the
    original `.` (id 236761), a training/inference alignment bug).

    If `snap_to_word_boundary=True` (default, 2026-08-19), every commit
    position is snapped forward to the next `▁`-prefixed token. Streaming
    inference can only fire commits at whitespace-word boundaries, so
    mid-word commits produce training rows the model can never reproduce
    at test time. Snapping aligns the training signal with what inference
    can actually query.
    """
    if not commit:
        return [], [], [], []
    # Consecutive-run grouping over commit[].
    groups: List[tuple[int, int, int]] = []  # (commit_i, tgt_start, tgt_end)
    j = 0
    while j < len(commit):
        k = j
        while k + 1 < len(commit) and commit[k + 1] == commit[j]:
            k += 1
        groups.append((commit[j], j, k))
        j = k + 1

    source_chunks, target_chunks = [], []
    source_chunk_ids, target_chunk_ids = [], []
    prev_src_end = 0
    for commit_i, jstart, jend in groups:
        if snap_to_word_boundary:
            commit_i = _snap_to_word_boundary(commit_i, src_token_ids, tokenizer, src_lang=src_lang)
            if commit_i <= prev_src_end:
                # Snap collapsed this commit into the previous one — merge
                # the target span into the previous group instead of emitting
                # an empty source chunk.
                if target_chunk_ids:
                    tgt_span = tgt_token_ids[jstart : jend + 1]
                    target_chunk_ids[-1] = target_chunk_ids[-1] + list(tgt_span)
                    target_chunks[-1] = (target_chunks[-1] + " "
                                          + tokenizer.decode(tgt_span, skip_special_tokens=True).strip()).strip()
                continue
        src_span = src_token_ids[prev_src_end:commit_i]
        tgt_span = tgt_token_ids[jstart : jend + 1]
        prev_src_end = commit_i
        source_chunks.append(tokenizer.decode(src_span, skip_special_tokens=True).strip())
        target_chunks.append(tokenizer.decode(tgt_span, skip_special_tokens=True).strip())
        source_chunk_ids.append(list(src_span))
        target_chunk_ids.append(list(tgt_span))

    # If the last commit did not exhaust the source, glue the tail onto the
    # final source chunk — the model must have "read" the whole source by
    # the time the sentence ends.
    if prev_src_end < n_src_tok:
        tail = src_token_ids[prev_src_end:n_src_tok]
        tail_str = tokenizer.decode(tail, skip_special_tokens=True).strip()
        if tail_str:
            if source_chunks:
                source_chunks[-1] = (source_chunks[-1] + " " + tail_str).strip()
                source_chunk_ids[-1] = source_chunk_ids[-1] + list(tail)
            else:
                source_chunks.append(tail_str)
                target_chunks.append("")
                source_chunk_ids.append(list(tail))
                target_chunk_ids.append([])

    return source_chunks, target_chunks, source_chunk_ids, target_chunk_ids


@torch.no_grad()
def annotate_pair(
    model,
    tokenizer,
    source: str,
    target: str,
    src_lang: str = "German",
    tgt_lang: str = "English",
    latency: str = "medium",
    tau: float = 0.05,
    criterion_name: str = "js",
    prefix_stride: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    return_full_matrix: bool = False,
    record_entropy: bool = False,
    prompt_mode: str = "raw",
    lookahead_k: int = 0,
) -> AnnotatedPair:
    """Annotate one parallel pair. Returns the AnnotatedPair with EAST-
    interleaved string populated.

    `prefix_stride > 1` evaluates every k-th source token — a coarse speed
    knob from METHOD §7. Commit points snap to the evaluated grid; keep
    stride=1 for the smoke.

    `return_full_matrix=True` populates `AnnotatedPair.divergence_matrix`
    with the full (n)-by-m divergence values D(P_full[j], P_pre[i][j])
    for i in 1..n (index 0..n-1 in the matrix) — enables offline tau
    sweeps from a single forward-pass run.

    `record_entropy=True` additionally populates `entropy_matrix` and
    `entropy_full` — for the entropy-only ablation criterion (METHOD §3).

    `prompt_mode` selects the input template:
      "raw"  — `{source_prefix}\\n{target}` (raw concat, no instruction).
      "chat" — Gemma-style chat template with an explicit translation
               instruction (`make_prompt_chat`). Use for instruction-
               tuned backbones like `gemma-4-E2B-it`.

    `lookahead_k` selects the reference distribution for the divergence:
      k = 0 (default) — current behaviour, D(P_full, P_pre[i]).
      k >= 1          — D(P_pre[i], P_pre[min(i+k, n)]); commit when
                        reading k more source tokens does not shift the
                        target distribution. Trailing positions (where
                        i + k >= n) fall back to P_full = P_pre[n].
    Negative values are clamped to 0.
    """
    lookahead_k = max(0, int(lookahead_k))
    if criterion_name == "ot":
        # Bind the model's input embeddings for the OT ground cost.
        emb = model.get_input_embeddings().weight
        divergence = make_ot(embedding_matrix=emb, topk=128, eps=0.05, sinkhorn_iters=200)
    elif criterion_name in CRITERIA:
        divergence = CRITERIA[criterion_name]
    else:
        raise ValueError(f"unknown criterion {criterion_name!r}; have {list(CRITERIA)} + 'ot'")
    if prompt_mode not in {"raw", "chat"}:
        raise ValueError(f"unknown prompt_mode {prompt_mode!r}; have {{raw, chat}}")

    if device is None:
        device = next(model.parameters()).device

    # Source and target are token slices; the prompt around them is built
    # at the string level (may vary by mode) and tokenised per prefix.
    src_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    n = len(src_ids)  # source token count
    m = len(tgt_ids)  # target token count
    if n == 0 or m == 0:
        raise ValueError("empty source or target after tokenisation")

    def _prompt_ids(i_src: int) -> List[int]:
        """Build the prompt tokens (everything before the target) for
        source prefix length `i_src`. Includes the chat template's BOS
        when prompt_mode='chat'; adds a manual BOS in raw mode."""
        src_prefix_str = tokenizer.decode(src_ids[:i_src], skip_special_tokens=True)
        if prompt_mode == "chat":
            # apply_chat_template already includes BOS for Gemma-family.
            prompt_str = make_prompt_chat(tokenizer, src_prefix_str, src_lang, tgt_lang)
            ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            return ids
        # raw
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        prompt_str = make_prompt_raw(src_prefix_str, src_lang, tgt_lang)
        ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        return bos + ids

    def build_input(i_src: int) -> Tuple[torch.Tensor, int]:
        prompt = _prompt_ids(i_src)
        ids = prompt + tgt_ids
        return torch.tensor([ids], device=device, dtype=torch.long), len(prompt)

    def target_positions_from_prefix_len(prefix_len: int) -> torch.Tensor:
        # The logit at position p predicts the token at p+1. Target token
        # j (0-indexed) sits at input position (prefix_len + j), predicted
        # by the logit at (prefix_len + j - 1).
        return torch.arange(prefix_len - 1, prefix_len - 1 + m, device=device)

    # P_full[j] with full source.
    full_input, full_prefix_len = build_input(n)
    full_positions = target_positions_from_prefix_len(full_prefix_len)
    p_full = _prob_at_positions(model, full_input, full_positions)  # (m, V)
    ent_full = _entropy(p_full).detach().cpu().tolist() if record_entropy else None

    # Commit point search per target token, sweeping i = 1..n on the given stride.
    commit = [n] * m  # fallback: if criterion never fires, commit at end
    fired_div = [float("inf")] * m
    active = list(range(m))  # target-token indices still searching
    # Full (n, m) matrix — row i-1 = divergence at source-prefix length i.
    # For k=0: div = D(P_full, P_pre[i]).
    # For k>0: div = D(P_pre[i], P_pre[min(i+k, n)]).
    full_matrix: List[List[float]] = []
    entropy_matrix: List[List[float]] = []
    if return_full_matrix:
        full_matrix = [[float("nan")] * m for _ in range(n)]
    if record_entropy:
        entropy_matrix = [[float("nan")] * m for _ in range(n)]

    # Ring buffer of source-prefix distributions, used only when lookahead_k > 0.
    # Holds at most k+1 entries; entry evicted immediately after its decision fires.
    prefix_probs_ring: dict[int, torch.Tensor] = {}

    def _apply_decision(position: int, div_tensor: torch.Tensor) -> None:
        """Commit-search + optional matrix write for a divergence row at
        source-prefix length `position` (1-indexed)."""
        div_cpu = div_tensor.detach().cpu()
        if return_full_matrix:
            full_matrix[position - 1] = [float(x) for x in div_cpu.tolist()]
        still = []
        for j in active:
            d = float(div_cpu[j].item())
            if d < tau:
                commit[j] = position
                fired_div[j] = d
            else:
                still.append(j)
        active[:] = still

    for i in range(1, n + 1):
        # Early-break is safe only when there is no delayed lookahead work
        # pending — with k>0 we must reach i=n so trailing decisions can fire.
        if (
            not active
            and not return_full_matrix
            and not record_entropy
            and lookahead_k == 0
        ):
            break
        if i != n and ((i - 1) % prefix_stride != 0):
            continue

        pre_input, pre_prefix_len = build_input(i)
        pre_positions = target_positions_from_prefix_len(pre_prefix_len)
        p_pre = _prob_at_positions(model, pre_input, pre_positions)  # (m, V)

        if record_entropy:
            ent = _entropy(p_pre).detach().cpu().tolist()
            entropy_matrix[i - 1] = [float(x) for x in ent]

        if lookahead_k == 0:
            div = divergence(p_full, p_pre)  # (m,)
            _apply_decision(i, div)
        else:
            # Buffer this prefix's probs; decide for position (i - k) if it
            # has fully materialised — the natural reference at (i-k) is
            # exactly P_pre[i]. Guard on ring membership so prefix_stride > 1
            # (which skips iterations) never KeyErrors.
            prefix_probs_ring[i] = p_pre
            decision_i = i - lookahead_k
            if decision_i >= 1 and decision_i in prefix_probs_ring:
                div = divergence(prefix_probs_ring[decision_i], p_pre)
                _apply_decision(decision_i, div)
                del prefix_probs_ring[decision_i]

        if verbose:
            print(f"  i={i:>3d}/{n}  active_left={len(active)}")

    # Trailing decisions for k>0: positions in {n-k+1, ..., n} still sit in the
    # ring buffer. Their reference should be P_pre[min(pos+k, n)] = P_pre[n],
    # which is exactly p_full.
    if lookahead_k > 0 and prefix_probs_ring:
        for decision_i in sorted(prefix_probs_ring.keys()):
            div = divergence(prefix_probs_ring[decision_i], p_full)
            _apply_decision(decision_i, div)
        prefix_probs_ring.clear()

    # Monotonicity: i*[j] = max(i*[j], i*[j-1])
    commit_mono = _enforce_monotone(commit)

    source_chunks, target_chunks, _src_ids, _tgt_ids = _chunks_from_commit(
        commit_mono, src_ids, tgt_ids, tokenizer, n, src_lang=src_lang
    )

    ann = AnnotatedPair(
        source=source,
        target=target,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        latency=latency,
        commit_source_tok_idx=commit_mono,
        fired_divergence=fired_div,
        source_chunks=source_chunks,
        target_chunks=target_chunks,
        n_src_tok=n,
        n_tgt_tok=m,
    )
    if return_full_matrix:
        ann.divergence_matrix = full_matrix
    if record_entropy:
        ann.entropy_matrix = entropy_matrix
        ann.entropy_full = ent_full

    try:
        ann.east_str = interleave(ann.as_east_row())
    except ValueError as e:
        # Chunk-count mismatch can happen if the tail-glue produced an
        # empty target chunk. Log rather than crash — sanity checks below
        # will surface it.
        ann.east_str = f"[interleave failed: {e}]"

    return ann
