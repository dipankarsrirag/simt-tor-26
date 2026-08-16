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


def _chunks_from_commit(
    commit: List[int],
    src_token_ids: List[int],
    tgt_token_ids: List[int],
    tokenizer,
    n_src_tok: int,
) -> tuple[list[str], list[str]]:
    """Group consecutive target tokens sharing a commit point into write
    chunks, and pair each with the read span that just became available."""
    if not commit:
        return [], []
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
    prev_src_end = 0
    for commit_i, jstart, jend in groups:
        src_span = src_token_ids[prev_src_end:commit_i]
        tgt_span = tgt_token_ids[jstart : jend + 1]
        prev_src_end = commit_i
        source_chunks.append(tokenizer.decode(src_span, skip_special_tokens=True).strip())
        target_chunks.append(tokenizer.decode(tgt_span, skip_special_tokens=True).strip())

    # If the last commit did not exhaust the source, glue the tail onto the
    # final source chunk — the model must have "read" the whole source by
    # the time the sentence ends.
    if prev_src_end < n_src_tok:
        tail = src_token_ids[prev_src_end:n_src_tok]
        tail_str = tokenizer.decode(tail, skip_special_tokens=True).strip()
        if tail_str:
            if source_chunks:
                source_chunks[-1] = (source_chunks[-1] + " " + tail_str).strip()
            else:
                source_chunks.append(tail_str)
                target_chunks.append("")

    return source_chunks, target_chunks


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
    """
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
    # Optional: full (n, m) matrix, row i-1 = divergence at prefix length i.
    full_matrix: List[List[float]] = [] if return_full_matrix else []
    entropy_matrix: List[List[float]] = [] if record_entropy else []

    for i in range(1, n + 1):
        if not active and not return_full_matrix and not record_entropy:
            break
        if i != n and ((i - 1) % prefix_stride != 0):
            if return_full_matrix:
                full_matrix.append([float("nan")] * m)
            if record_entropy:
                entropy_matrix.append([float("nan")] * m)
            continue

        pre_input, pre_prefix_len = build_input(i)
        pre_positions = target_positions_from_prefix_len(pre_prefix_len)
        p_pre = _prob_at_positions(model, pre_input, pre_positions)  # (m, V)

        # Divergence at every j; check the currently-searching ones.
        div = divergence(p_full, p_pre)  # (m,)
        div_cpu = div.detach().cpu()
        if return_full_matrix:
            full_matrix.append([float(x) for x in div_cpu.tolist()])
        if record_entropy:
            ent = _entropy(p_pre).detach().cpu().tolist()
            entropy_matrix.append([float(x) for x in ent])
        still_active = []
        for j in active:
            d = float(div_cpu[j].item())
            if d < tau:
                commit[j] = i
                fired_div[j] = d
            else:
                still_active.append(j)
        active = still_active

        if verbose:
            print(f"  i={i:>3d}/{n}  active_left={len(active)}")

    # Monotonicity: i*[j] = max(i*[j], i*[j-1])
    commit_mono = _enforce_monotone(commit)

    source_chunks, target_chunks = _chunks_from_commit(
        commit_mono, src_ids, tgt_ids, tokenizer, n
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
