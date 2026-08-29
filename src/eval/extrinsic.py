"""
Extrinsic eval harness for Phase-2 SFT models (Gate 3 first cut).

Layers:
  1. offline   — full-source generation, BLEU only. Sanity: does the model
                 translate at all?
                 STATUS: landed. OT-SFT n=10K 32.54 on newstest2013.
  2. streaming — state-machine READ/WRITE with KV-cache preservation, per
                 EAST §3.2 inference protocol. Two policies:
                   - check_argmax: after each source word fed, poll model
                     argmax; if EOR, switch to WRITE. Model-driven policy.
                   - wait_k: force EOR after every k source words. AL unit
                     test (should land AL ≈ k on the corpus).
                 Tracks AL (Ma 2019) + LAAL (Papi 2022) at word units.
  3. AL-CA     — torch.cuda.Event per emitted target token; warmup discard.
                 (Punted from first cut — first-cut ships offline + streaming
                 BLEU + AL only. Corpus-level approximation available via
                 scripts/phase2_compute_al_ca_approx.py.)

Single-arm setup since 2026-08-18 late — this harness runs the OT-SFT
model (dir: `_archive/results/v6b_gemma_2b/sft_*/final`). Cond-A (GPT-4 chunks) and
Cond-C (within-framework wait-k baseline) were both removed after the
decision to compare against past-work published numbers verbatim rather
than run our own matched baselines. See `../LOG.md`
`[DECISION] 2026-08-18 late — Remove cond-A entirely` and
`[DECISION] 2026-08-18 late — Remove Cond-C`.

Uses the extended tokenizer with the 5 EAST special tokens
(`tokenizer-extended` for E2B; `tokenizer-e4b-extended` for E4B;
`tokenizer-qwen35-2b-extended` for Qwen). Tokenizer drift poisons every
downstream number — `--tokenizer_dir` is non-optional.

Dev-first discipline: report numbers on newstest2013. newstest2015 is the
test set and gets touched exactly once, after dev-set behaviour is
understood. WMT22 De→En optionally added for EAST Table-3 head-to-head.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch

from src.annotator.east_format import (
    END_OF_READ,
    END_OF_WRITE,
    LATENCY_TOKENS,
    SPECIAL_TOKENS,
)


# ─── I/O ───────────────────────────────────────────────────────────────

def load_dev_pairs(src_path: Path, ref_path: Path) -> List[Dict[str, str]]:
    """Load parallel .de/.en, one sentence per line. Requires 1:1 alignment."""
    src = src_path.read_text().splitlines()
    ref = ref_path.read_text().splitlines()
    if len(src) != len(ref):
        raise ValueError(f"src/ref line count mismatch: {len(src)} vs {len(ref)}")
    return [{"src": s, "ref": r} for s, r in zip(src, ref)]


# ─── Layer 1: offline ──────────────────────────────────────────────────

@torch.no_grad()
def generate_offline(
    model,
    tok,
    src: str,
    latency: str,
    max_new_tokens: int,
    device: str,
) -> str:
    """Full-source greedy generation as the degenerate "one giant chunk" case
    of the streaming protocol. Prompt closes the READ phase explicitly with
    <|end-of-read|> so the model sees a valid EAST training shape:

        <|latency|> src_1 ... src_n <|end-of-read|>  → generate target tokens

    (Without the trailing EOR, the input `<|latency|> src` is not a shape the
    model ever saw during SFT — behaviour there is undefined and BLEU is not
    interpretable. cond-B in particular has real single-chunk training rows
    where the whole source is one READ.)

    Returns the raw decoded string with all EAST special tokens + BOS/EOS
    stripped by `skip_special_tokens=True`.
    """
    prompt = f"{LATENCY_TOKENS[latency]} {src} {END_OF_READ}"
    inp = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_len = inp.shape[1]
    eow_id = tok(END_OF_WRITE, add_special_tokens=False).input_ids[0]
    out = model.generate(
        input_ids=inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=[tok.eos_token_id, eow_id],
    )
    gen_ids = out[0][prompt_len:].tolist()
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


# ─── Layer 2: streaming ────────────────────────────────────────────────

def tokenize_source_by_words(tok, src: str, src_lang: str = "en") -> Tuple[List[int], List[List[int]]]:
    """Tokenize source and map BPE tokens to whitespace-word spans.

    2026-08-22 v6b fix: word[0] tokenized WITHOUT leading space; word[i>0]
    tokenized WITH leading space. This matches the annotator's full-source
    tokenization (`tok(src_clean)`), which is now what the v6 SFT dataset
    stores as source_chunk_ids (and what direct-ids-splice training feeds
    to the model verbatim). So streaming inference produces the same ids
    the model saw during training.

    Prior v5 behavior (prepend space to EVERY word, including word[0])
    was correct for the string-round-trip training path where the source
    was embedded as ` source_chunk` after `<|latency|>` — the leading
    space imputed the first `▁`. v6 no longer uses that path.

    For CJK-family languages (zh/ja/ko/th/km) there is no whitespace in the
    source. `src.split()` would return `[src]` (one giant "word"). Split
    by CHARACTER and tokenize each without a leading space — matches the
    annotator's no-leading-space treatment for CJK.

    Returns:
      full_ids: the token IDs for the source (byte-identical to `tok(src)`).
      word_spans: list of len(words), each entry is the list of BPE token
                  IDs belonging to that whitespace-word (or character for CJK).
    """
    from src.annotator.annotate import _is_cjk_lang  # local import to avoid cycles
    is_cjk = _is_cjk_lang(src_lang)

    if is_cjk:
        # CJK: split by character. Each char is its own "word" for streaming.
        # No leading space — training-time CJK src is fed without ` ` prefix.
        src_clean = src.replace(" ", "")  # drop any incidental whitespace
        chars = list(src_clean)
        spans = []
        for ch in chars:
            ids = tok(ch, add_special_tokens=False).input_ids
            spans.append(ids)
        naive_ids = [t for s in spans for t in s]
        full_ids = tok(src_clean, add_special_tokens=False).input_ids
        if naive_ids == full_ids:
            return full_ids, spans
        # Fallback: attribute each BPE to a char span via offset_mapping.
        # For CJK this rarely differs (mostly 1 BPE per char), but be safe.
        enc = tok(src_clean, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = enc.input_ids
        offsets = enc.offset_mapping
        spans = [[] for _ in chars]
        for tok_id, (a, b) in zip(full_ids, offsets):
            ci = a if a < len(src_clean) else max(0, len(src_clean) - 1)
            spans[min(ci, len(chars) - 1)].append(tok_id)
        return full_ids, spans

    words = src.split()
    # word[0]: no leading space (first token gets no `▁`); word[i>0]: leading
    # space (subsequent word first tokens get `▁`). Matches annotator's
    # `tok(src_clean)`.
    spans = []
    for wi, w in enumerate(words):
        prefix = w if wi == 0 else " " + w
        ids = tok(prefix, add_special_tokens=False).input_ids
        spans.append(ids)
    naive_ids = [t for s in spans for t in s]
    # Cross-check: concatenated per-word ids should equal full-source
    # tokenization. If not (rare boundary-merge cases), fall through to
    # offset_mapping-based attribution.
    full_ids = tok(src, add_special_tokens=False).input_ids
    if naive_ids == full_ids:
        return full_ids, spans
    # Fallback: use offset_mapping on `src` directly.
    enc = tok(src, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = enc.input_ids
    offsets = enc.offset_mapping
    word_of_char = [-1] * len(src)
    ci = 0
    for wi, w in enumerate(words):
        while ci < len(src) and src[ci].isspace():
            ci += 1
        for _ in range(len(w)):
            if ci < len(src):
                word_of_char[ci] = wi
                ci += 1
    spans = [[] for _ in words]
    for tok_id, (a, b) in zip(full_ids, offsets):
        if b <= a:
            wi = 0 if not words else 0
        else:
            mid = (a + b) // 2
            wi = word_of_char[mid] if mid < len(word_of_char) else -1
            if wi < 0:
                wi = 0
        spans[wi].append(tok_id)
    return full_ids, spans


@dataclass
class StreamTrace:
    """Per-sentence streaming trace (compact record for debugging + AL)."""
    src_words: int
    tgt_word_g: List[int] = field(default_factory=list)  # g_words(i) per target word
    chunks_committed: int = 0
    source_exhausted_without_eor: bool = False
    write_cap_hits: int = 0
    hyp: str = ""


@torch.no_grad()
def stream_translate(
    model,
    tok,
    src: str,
    latency: str,
    device: str,
    policy: str = "check_argmax",   # see below
    wait_k: int = 3,
    max_write_per_chunk: int = 40,
    commit_prob_thresh: float = 0.10,
    commit_rank: int = 3,
    commit_ratio: float = 0.5,
    src_lang: str = "en",
    tgt_lang: str = "en",
    use_chat_template: bool = False,
) -> StreamTrace:
    """Streaming inference. Feeds source words one at a time, maintains a
    KV cache, and lets one of several policies drive commit points.

    Hard-argmax policies (baseline):
      - check_argmax : commit if argmax(logits) == EOR
      - wait_k       : commit every k source words (deterministic schedule)

    Soft-commit adaptivity probes (Test A per docs/next-steps.md — asks
    whether learned adaptivity is hidden behind hard argmax):
      - check_prob_thresh : commit if p(EOR) > commit_prob_thresh
      - check_rank        : commit if rank(EOR) <= commit_rank
      - check_ratio       : commit if p(EOR) / p(top_non_eor) > commit_ratio
                            (top_non_eor = argmax over vocab minus EOR — the
                            model's best continuation guess if it did NOT
                            want to commit; typically the next German subword)

    Word-unit AL bookkeeping (Ma 2019 §4):
      g_words(i) = number of source words fully read when target word i is
                   emitted. Piecewise-constant within a chunk (source read
                   pauses during WRITE).
    """
    # Tokenizers + special-token IDs.
    eor_id = tok(END_OF_READ, add_special_tokens=False).input_ids[0]
    eow_id = tok(END_OF_WRITE, add_special_tokens=False).input_ids[0]
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id

    # Byte-identical source token sequence, grouped by word (or per-char for CJK).
    _, src_word_spans = tokenize_source_by_words(tok, src, src_lang=src_lang)
    n_src_words = len(src_word_spans)

    trace = StreamTrace(src_words=n_src_words)

    # Feed initial prompt.
    if use_chat_template:
        # v6 chat prompt: system + user instruction + generation-prompt.
        # The user turn carries direction (src→tgt language names) and
        # latency (natural-language 5-point ladder: low, low-medium, medium,
        # medium-high, high). Model has been SFT'd to emit EOR/EOW-interleaved
        # translation as the assistant turn.
        from src.annotator.east_format import (
            build_user_instruction, DEFAULT_SYSTEM_PROMPT
        )
        user_instr = build_user_instruction(src_lang, tgt_lang, latency)
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_instr},
        ]
        try:
            prompt_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: merge system into user
            messages = [{"role": "user",
                          "content": DEFAULT_SYSTEM_PROMPT + "\n\n" + user_instr}]
            prompt_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompt_ids = tok(prompt_str, add_special_tokens=False).input_ids
    else:
        # v1-v5 path: [BOS] <|latency|>
        latency_id = tok(LATENCY_TOKENS[latency], add_special_tokens=False).input_ids[0]
        prompt_ids = [bos_id, latency_id] if bos_id is not None else [latency_id]
    input_ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values

    src_words_read = 0
    tgt_ids_all: List[int] = []
    tgt_chunk_start_indices: List[int] = []  # tgt_ids_all index where each chunk starts
    chunk_g_words: List[int] = []            # g_words at start of each chunk

    def feed(ids: List[int]):
        """Append tokens to the model, update KV. Returns final-position logits."""
        nonlocal kv
        t = torch.tensor([ids], device=device)
        o = model(input_ids=t, past_key_values=kv, use_cache=True)
        kv = o.past_key_values
        return o.logits[0, -1, :]

    def generate_write_chunk(g_at_start: int):
        """Generate target tokens greedily until EOW, EOS, EOR, or per-chunk
        cap. Model-emitted EOR mid-write is treated as end-of-chunk (model
        wants source that the caller can force-feed on next outer iteration);
        without this the model hallucinates German "source" text mid-target
        after emitting EOR, because training pattern is
        `<eor> tgt <eow> <src> <eor> tgt <eow>...` and the model completes it.
        Returns (n_tokens_generated, hit_cap)."""
        nonlocal tgt_ids_all
        n = 0
        # After feed([eor]), the returned logits are for the NEXT position —
        # the first target token.
        prev_logits = feed([eor_id])
        while n < max_write_per_chunk:
            next_id = int(prev_logits.argmax().item())
            if next_id == eow_id:
                feed([eow_id])
                return n, False
            if next_id == eos_id:
                feed([eos_id])
                return n, False
            if next_id == eor_id:
                # Model wants a new READ chunk mid-write. Terminate this
                # chunk; caller decides whether to feed more source or
                # force final EOR again.
                return n, False
            tgt_ids_all.append(next_id)
            n += 1
            prev_logits = feed([next_id])
        return n, True  # cap hit

    # ── main streaming loop ────────────────────────────────
    for wi in range(n_src_words):
        span = src_word_spans[wi]
        if not span:
            src_words_read += 1
            continue
        # Feed this word's BPE tokens.
        logits = feed(span)
        src_words_read += 1
        # Decide commit based on policy.
        if policy == "wait_k":
            commit = (src_words_read % wait_k == 0)
        elif policy == "check_argmax":
            commit = (int(logits.argmax().item()) == eor_id)
        elif policy == "check_prob_thresh":
            p = torch.softmax(logits.float(), dim=-1)
            commit = (float(p[eor_id].item()) > commit_prob_thresh)
        elif policy == "check_rank":
            # rank(EOR) among all vocab positions; rank 1 == argmax.
            eor_logit = logits[eor_id]
            rank = int((logits > eor_logit).sum().item()) + 1
            commit = (rank <= commit_rank)
        elif policy == "check_ratio":
            p = torch.softmax(logits.float(), dim=-1)
            p_eor = float(p[eor_id].item())
            # top non-EOR probability: temporarily mask EOR, take argmax.
            p_masked = p.clone()
            p_masked[eor_id] = 0.0
            p_top_non_eor = float(p_masked.max().item())
            if p_top_non_eor <= 0.0:
                commit = True
            else:
                commit = ((p_eor / p_top_non_eor) > commit_ratio)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        if commit:
            # Enter WRITE mode: consume EOR (its logits predict first target token).
            tgt_chunk_start_indices.append(len(tgt_ids_all))
            chunk_g_words.append(src_words_read)
            n_gen, hit_cap = generate_write_chunk(g_at_start=src_words_read)
            trace.chunks_committed += 1
            if hit_cap:
                trace.write_cap_hits += 1

    # Source exhausted. Force a final EOR ONLY if the last committed chunk
    # was at an earlier src position — otherwise `<eow><eor>` back-to-back
    # is a pattern the model NEVER saw in training (training format is
    # `<eow> src <eor>` with source in between) and it hallucinates a
    # German source chunk in the drain output.
    last_commit_at = chunk_g_words[-1] if chunk_g_words else -1
    need_drain = (trace.chunks_committed == 0) or (last_commit_at < src_words_read)
    if need_drain:
        if trace.chunks_committed == 0:
            trace.source_exhausted_without_eor = True
        tgt_chunk_start_indices.append(len(tgt_ids_all))
        chunk_g_words.append(src_words_read)
        n_gen, hit_cap = generate_write_chunk(g_at_start=src_words_read)
        trace.chunks_committed += 1
        if hit_cap:
            trace.write_cap_hits += 1

    # Decode the concatenated target tokens.
    hyp = tok.decode(tgt_ids_all, skip_special_tokens=True).strip()
    trace.hyp = hyp

    # AL bookkeeping in WORD units. Split decoded hyp on whitespace to get target
    # words. Attribute g_words to each target word by which chunk it fell in.
    # Approximation: split each chunk's decoded string into words, tag all with
    # that chunk's g_at_start. Robust to BPE internal boundaries.
    tgt_word_g: List[int] = []
    for ci in range(len(tgt_chunk_start_indices)):
        start = tgt_chunk_start_indices[ci]
        end = tgt_chunk_start_indices[ci + 1] if ci + 1 < len(tgt_chunk_start_indices) else len(tgt_ids_all)
        chunk_ids = tgt_ids_all[start:end]
        chunk_str = tok.decode(chunk_ids, skip_special_tokens=True).strip()
        n_words = len(chunk_str.split())
        tgt_word_g.extend([chunk_g_words[ci]] * n_words)
    trace.tgt_word_g = tgt_word_g
    return trace


def compute_al(g_words: List[int], x_len: int, y_len: int) -> Optional[float]:
    """AL (Average Lagging) per Ma et al. 2019 §4.

    AL(g) = (1/tau) * sum_{i=1..tau} (g(i) - (i-1) * |X| / |Y|)
    tau = argmin_i (g(i) = |X|)   [first i where all source has been read]

    Returns None if the alignment is degenerate (y_len==0, or model never
    reads all source before target ends).
    """
    if y_len == 0 or x_len == 0:
        return None
    tau = None
    for i, g in enumerate(g_words, start=1):
        if g >= x_len:
            tau = i
            break
    if tau is None:
        # Never fully read source by the time target ended — degenerate. Use
        # tau = y_len as a fallback (Ma 2019 convention for such cases).
        tau = len(g_words)
    if tau == 0:
        return None
    ratio = x_len / y_len
    s = 0.0
    for i in range(1, tau + 1):
        g = g_words[i - 1]
        s += g - (i - 1) * ratio
    return s / tau


def compute_laal(g_words: List[int], x_len: int, y_len: int) -> Optional[float]:
    """LAAL (Length-Adaptive Average Lagging) per Papi et al. 2022.

    LAAL(g) = (1/|Y|) * sum_{i=1..|Y|} (g(i) - (i-1) * |X| / |Y|)

    Same numerator terms as AL but summed over ALL target tokens (no
    truncation at source-exhaustion). Handles over-generation gracefully —
    target tokens emitted after g(i)=|X| still contribute their lag.
    Typically LAAL >= AL; delta is 0.2-1.5 source-word-equivalents on WMT
    De->En streaming outputs (Papi et al. 2022).

    Reported alongside AL so competitor numbers using either variant can
    be compared to ours — see _archive/docs/phase2-sft-and-streaming.md
    'Cross-paper comparability protocol'.
    """
    if y_len == 0 or x_len == 0 or not g_words:
        return None
    ratio = x_len / y_len
    n = len(g_words)
    s = 0.0
    for i in range(1, n + 1):
        g = g_words[i - 1]
        s += g - (i - 1) * ratio
    return s / n


# ─── Runner ────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    model_dir: str
    tokenizer_dir: str
    dev_src: str
    dev_ref: str
    latency: str
    n_sentences: int
    max_new_tokens: int
    mode: str              # offline | streaming
    policy: str            # check_argmax | wait_k | check_prob_thresh | check_rank | check_ratio
    wait_k: int            # k (streaming wait_k policy only)
    max_write_per_chunk: int
    output: str
    commit_prob_thresh: float = 0.10
    commit_rank: int = 3
    commit_ratio: float = 0.5
    src_lang: str = "en"
    tgt_lang: str = "en"
    use_chat_template: bool = False


def run(cfg: RunConfig) -> Dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import sacrebleu

    print(f"[extrinsic] mode={cfg.mode} latency={cfg.latency}", flush=True)
    if cfg.mode == "streaming":
        print(f"[extrinsic] policy={cfg.policy} (k={cfg.wait_k}) max_write_per_chunk={cfg.max_write_per_chunk}", flush=True)
    print(f"[extrinsic] loading tokenizer {cfg.tokenizer_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    print(f"[extrinsic] loading model {cfg.model_dir}", flush=True)
    cfg_hf = AutoConfig.from_pretrained(cfg.model_dir)
    if getattr(cfg_hf, "model_type", None) == "gemma3n":
        from transformers import Gemma3nForCausalLM
        print("[extrinsic] (model_type=gemma3n; loading text-only Gemma3nForCausalLM)", flush=True)
        model = Gemma3nForCausalLM.from_pretrained(cfg.model_dir, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # v6 tokenizer only has EOR + EOW added (latency is NL, not vocab tokens).
    # v1-v5 tokenizer has EOR + EOW + 3 latency tokens.
    from src.annotator.east_format import SPECIAL_TOKENS_V6
    check_tokens = SPECIAL_TOKENS_V6 if cfg.use_chat_template else SPECIAL_TOKENS
    for t in check_tokens:
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"special token {t!r} splits into {len(ids)} ids — tokenizer drift")

    pairs = load_dev_pairs(Path(cfg.dev_src), Path(cfg.dev_ref))
    if cfg.n_sentences > 0:
        pairs = pairs[: cfg.n_sentences]
    print(f"[extrinsic] scoring {len(pairs)} sentences", flush=True)

    hyps: List[str] = []
    refs: List[str] = [p["ref"] for p in pairs]
    al_values: List[float] = []
    laal_values: List[float] = []
    stream_stats = {
        "n_source_exhausted_without_eor": 0,
        "n_write_cap_hits": 0,
        "chunk_counts": [],
    }
    t0 = time.time()
    for i, p in enumerate(pairs):
        if cfg.mode == "offline":
            h = generate_offline(model, tok, p["src"], cfg.latency,
                                 cfg.max_new_tokens, device)
            hyps.append(h)
        elif cfg.mode == "streaming":
            trace = stream_translate(
                model, tok, p["src"], cfg.latency, device,
                policy=cfg.policy, wait_k=cfg.wait_k,
                max_write_per_chunk=cfg.max_write_per_chunk,
                commit_prob_thresh=cfg.commit_prob_thresh,
                commit_rank=cfg.commit_rank,
                commit_ratio=cfg.commit_ratio,
                src_lang=cfg.src_lang,
                tgt_lang=cfg.tgt_lang,
                use_chat_template=cfg.use_chat_template,
            )
            hyps.append(trace.hyp)
            # AL uses self-consistent y_len = len(tgt_word_g). Using
            # hyp.split() length here would mismatch g_list length whenever
            # BPE splits a word across a chunk boundary (~5-10% inflation),
            # yielding an artificially small AL.
            y_len_g = len(trace.tgt_word_g)
            al = compute_al(trace.tgt_word_g, trace.src_words, y_len_g)
            laal = compute_laal(trace.tgt_word_g, trace.src_words, y_len_g)
            if al is not None:
                al_values.append(al)
            if laal is not None:
                laal_values.append(laal)
            if trace.source_exhausted_without_eor:
                stream_stats["n_source_exhausted_without_eor"] += 1
            stream_stats["n_write_cap_hits"] += trace.write_cap_hits
            stream_stats["chunk_counts"].append(trace.chunks_committed)
            stream_stats.setdefault("per_sent", []).append({
                "idx": i, "src_words": trace.src_words,
                "y_len_g": y_len_g, "y_len_hyp": len(trace.hyp.split()),
                "chunks": trace.chunks_committed,
                "al": al, "laal": laal, "g_words": trace.tgt_word_g,
            })
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")

        if i < 3 or (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(pairs)}] src={p['src'][:70]!r}", flush=True)
            print(f"           hyp={hyps[-1][:140]!r}", flush=True)
            if cfg.mode == "streaming" and al_values:
                import statistics as s
                print(f"           corpus AL so far: mean={s.mean(al_values):.2f}  n={len(al_values)}", flush=True)
    wall = time.time() - t0
    print(f"[extrinsic] generation wall: {wall:.1f}s ({wall/len(pairs):.2f}s/sent)", flush=True)

    metric = sacrebleu.BLEU()
    bleu = metric.corpus_score(hyps, [refs])
    signature = str(metric.get_signature())
    print(f"[extrinsic] BLEU = {bleu.score:.2f} ({signature})", flush=True)

    result = {
        "config": asdict(cfg),
        "n_sentences": len(pairs),
        "bleu": bleu.score,
        "bleu_signature": signature,
        "wall_time_sec": wall,
        "sec_per_sentence": wall / len(pairs),
        "hyps": hyps,
        "refs": refs,
    }
    if cfg.mode == "streaming":
        import statistics as s
        result["al_mean"] = s.mean(al_values) if al_values else None
        result["al_median"] = s.median(al_values) if al_values else None
        result["al_n_defined"] = len(al_values)
        result["laal_mean"] = s.mean(laal_values) if laal_values else None
        result["laal_median"] = s.median(laal_values) if laal_values else None
        result["laal_n_defined"] = len(laal_values)
        result["stream_stats"] = {
            "n_source_exhausted_without_eor": stream_stats["n_source_exhausted_without_eor"],
            "n_write_cap_hits": stream_stats["n_write_cap_hits"],
            "chunks_per_sent_mean": s.mean(stream_stats["chunk_counts"]),
            "chunks_per_sent_median": s.median(stream_stats["chunk_counts"]),
            "per_sent": stream_stats.get("per_sent", []),
        }
        print(f"[extrinsic] AL   mean={result['al_mean']:.2f}  median={result['al_median']:.2f}  n_defined={result['al_n_defined']}/{len(pairs)}", flush=True)
        if result["laal_mean"] is not None:
            print(f"[extrinsic] LAAL mean={result['laal_mean']:.2f}  median={result['laal_median']:.2f}  n_defined={result['laal_n_defined']}/{len(pairs)}", flush=True)
        print(f"[extrinsic] chunks/sent mean={result['stream_stats']['chunks_per_sent_mean']:.2f}  median={result['stream_stats']['chunks_per_sent_median']:.1f}", flush=True)
        print(f"[extrinsic] source-exhausted-without-eor: {stream_stats['n_source_exhausted_without_eor']}/{len(pairs)}", flush=True)
        print(f"[extrinsic] write-cap hits: {stream_stats['n_write_cap_hits']}", flush=True)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    # Retry on transient NCI /g/data filesystem I/O errors (Errno 5). Observed
    # on 2026-08-22 low+medium full-FLORES jobs — cost 30 min GPU each.
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    for attempt in range(5):
        try:
            Path(cfg.output).write_text(payload)
            print(f"[extrinsic] wrote {cfg.output}", flush=True)
            break
        except OSError as e:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16s
            print(f"[extrinsic] write attempt {attempt+1}/5 failed ({e!r}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        # All 5 retries failed; dump to jobfs as last resort so eval isn't lost.
        import os as _os
        fallback = f"{_os.environ.get('PBS_JOBFS', '/tmp')}/{Path(cfg.output).name}"
        Path(fallback).write_text(payload)
        print(f"[extrinsic] FALLBACK: wrote {fallback} (target /g/data still failing)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="e.g. _archive/results/v6b_gemma_2b/sft_n10k/final (OT-SFT, our method).")
    ap.add_argument("--tokenizer_dir",
                    default="/g/data/ba39/dipankar/simt-tor-26/_archive/results/v6b_gemma_2b/tokenizer-extended")
    ap.add_argument("--dev_src", required=True)
    ap.add_argument("--dev_ref", required=True)
    ap.add_argument("--latency", type=str, default="medium",
                    help="Latency prompt. v1-v5: {low,medium,high}. v6 chat-template: "
                         "{low, low-medium, medium, medium-high, high} (5-point NL ladder).")
    ap.add_argument("--n_sentences", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=200,
                    help="Offline mode: cap on decoded target tokens.")
    ap.add_argument("--mode", choices=["offline", "streaming"], default="offline")
    ap.add_argument("--policy",
                    choices=["check_argmax", "wait_k",
                             "check_prob_thresh", "check_rank", "check_ratio"],
                    default="check_argmax",
                    help="Streaming policy. wait_k is the AL unit test; "
                         "check_prob_thresh/check_rank/check_ratio are Test A "
                         "soft-commit adaptivity probes.")
    ap.add_argument("--wait_k", type=int, default=3)
    ap.add_argument("--commit_prob_thresh", type=float, default=0.10,
                    help="check_prob_thresh: commit if p(EOR) > this.")
    ap.add_argument("--commit_rank", type=int, default=3,
                    help="check_rank: commit if rank(EOR) <= this.")
    ap.add_argument("--commit_ratio", type=float, default=0.5,
                    help="check_ratio: commit if p(EOR)/p(top_non_eor) > this.")
    ap.add_argument("--max_write_per_chunk", type=int, default=40,
                    help="Streaming: cap on target tokens per WRITE chunk before "
                         "forcing return to READ. If >5%% of chunks hit this, "
                         "the WRITE-stop mechanism is broken.")
    ap.add_argument("--src_lang", type=str, default="en",
                    help="Source language code (en/de/ar/ru/zh/vi/...). Used for CJK "
                         "streaming routing + v6 chat-template direction phrase.")
    ap.add_argument("--tgt_lang", type=str, default="en",
                    help="Target language code (v6 chat-template only, for direction phrase).")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="v6 mode: apply Gemma chat template with NL translation instruction. "
                         "Requires an instruct-tuned backbone + tokenizer-extended-v6.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = RunConfig(
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        dev_src=args.dev_src,
        dev_ref=args.dev_ref,
        latency=args.latency,
        n_sentences=args.n_sentences,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        policy=args.policy,
        wait_k=args.wait_k,
        max_write_per_chunk=args.max_write_per_chunk,
        output=args.output,
        commit_prob_thresh=args.commit_prob_thresh,
        commit_rank=args.commit_rank,
        commit_ratio=args.commit_ratio,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        use_chat_template=args.use_chat_template,
    )
    run(cfg)


if __name__ == "__main__":
    main()
