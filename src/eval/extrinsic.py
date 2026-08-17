"""
Extrinsic eval harness for Phase-2 SFT models (Gate 3 first cut).

Layers, mirroring the advisor spec (2026-08-17):
  1. offline   — full-source generation, BLEU only. Sanity: does the model
                 translate at all? Both A and B must have offline BLEU > 10
                 before streaming numbers are worth reading.
                 STATUS: landed. cond-A n=10K 32.41, cond-B n=10K 32.54 on
                 newstest2013.
  2. streaming — state-machine READ/WRITE with KV-cache preservation, per
                 EAST §3.2 inference protocol. Two policies:
                   - check_argmax: after each source word fed, poll model
                     argmax; if EOR, switch to WRITE. Model-driven policy.
                   - wait_k: force EOR after every k source words. AL unit
                     test (should land AL ≈ k on the corpus).
                 Tracks AL (word units, Ma 2019 §4).
  3. AL-CA     — torch.cuda.Event per emitted target token; warmup discard.
                 (Punted from first cut — first-cut ships offline + streaming
                 BLEU + AL only.)

The two conditions being compared:
  cond-A = trained on shipped GPT-4 chunks (SiMT-660K's source_chunks).
  cond-B = trained on our OT-annotator chunks (results/phase2/annot_ot_*).

Both use the SAME extended tokenizer (results/phase2/tokenizer-extended)
and the same 5 EAST special tokens at ids 262144-262148. Any drift here
poisons every downstream number — the tokenizer_dir CLI flag is
non-optional.

Dev-first discipline (advisor): report numbers on newstest2013 to validate
the pipeline. newstest2015 is the test set and gets touched exactly once,
after A-vs-B behaviour on the dev set is understood.
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

def tokenize_source_by_words(tok, src: str) -> Tuple[List[int], List[List[int]]]:
    """Tokenize source and map BPE tokens to whitespace-word spans.

    Critical to correctness (advisor 2026-08-17): tokenizing " " + word_i
    in a loop yields different IDs than tokenizing the full source in one
    shot, because SentencePiece's leading-space and cross-boundary BPE
    merges depend on context. Model was trained on the full-concatenated
    form; we must feed the identical token sequence, just paced word by
    word.

    Returns:
      full_ids: the token IDs for the full source (add_special_tokens=False).
      word_spans: list of len(words), each entry is the list of BPE token
                  IDs belonging to that whitespace-word.

    Approach: tokenize each word alone WITH a leading space (matches how
    SentencePiece would produce it mid-sentence), concatenate spans, and
    verify the concatenated IDs match tokenizing the full string. If they
    don't (leading-word edge case), fall back to walking the offsets.
    """
    words = src.split()
    # Naive first pass: " word_i" tokenized independently. First word has no
    # leading space at true sentence-start.
    spans = []
    for i, w in enumerate(words):
        prefix = w if i == 0 else " " + w
        ids = tok(prefix, add_special_tokens=False).input_ids
        spans.append(ids)
    naive_ids = [t for s in spans for t in s]
    # Cross-check: does the naive concat match a full-source tokenization?
    full_ids = tok(src, add_special_tokens=False).input_ids
    if naive_ids == full_ids:
        return full_ids, spans
    # Fallback: use offset_mapping to assign each BPE token to its word.
    # Some SentencePiece configs merge across whitespace; offsets are truth.
    enc = tok(src, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = enc.input_ids
    offsets = enc.offset_mapping
    # Build char-position -> word-index map.
    word_of_char = [-1] * len(src)
    ci = 0
    for wi, w in enumerate(words):
        # Skip whitespace before this word.
        while ci < len(src) and src[ci].isspace():
            ci += 1
        for _ in range(len(w)):
            if ci < len(src):
                word_of_char[ci] = wi
                ci += 1
    spans = [[] for _ in words]
    for tok_id, (a, b) in zip(full_ids, offsets):
        if b <= a:
            # Special tokens or SentencePiece phantom offsets — attach to the
            # last-seen word (or word 0 if none yet).
            wi = spans and (len(spans) - 1) if spans[-1] else 0
        else:
            # Use midpoint char to pick word.
            wi = word_of_char[(a + b) // 2]
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
    policy: str = "check_argmax",   # "check_argmax" | "wait_k"
    wait_k: int = 3,
    max_write_per_chunk: int = 40,
) -> StreamTrace:
    """Streaming inference. Feeds source words one at a time, maintains a
    KV cache, and lets either the model (check_argmax) or a fixed wait-k
    schedule drive commit points.

    Word-unit AL bookkeeping (Ma 2019 §4):
      g_words(i) = number of source words fully read when target word i is
                   emitted. Piecewise-constant within a chunk (source read
                   pauses during WRITE).
    """
    # Tokenizers + special-token IDs.
    eor_id = tok(END_OF_READ, add_special_tokens=False).input_ids[0]
    eow_id = tok(END_OF_WRITE, add_special_tokens=False).input_ids[0]
    latency_id = tok(LATENCY_TOKENS[latency], add_special_tokens=False).input_ids[0]
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id

    # Byte-identical source token sequence, grouped by word.
    _, src_word_spans = tokenize_source_by_words(tok, src)
    n_src_words = len(src_word_spans)

    trace = StreamTrace(src_words=n_src_words)

    # Feed initial prompt: [BOS] <|latency|>
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

    # Source exhausted. If no chunk emitted yet OR model still has target left,
    # force a final EOR and drain.
    if trace.chunks_committed == 0 or True:
        # Always force a final EOR to drain the remaining target — matches EAST
        # inference (source exhaust → one last WRITE covers the tail).
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
    policy: str            # check_argmax | wait_k (streaming only)
    wait_k: int            # k (streaming wait_k policy only)
    max_write_per_chunk: int
    output: str


def run(cfg: RunConfig) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sacrebleu

    print(f"[extrinsic] mode={cfg.mode} latency={cfg.latency}", flush=True)
    if cfg.mode == "streaming":
        print(f"[extrinsic] policy={cfg.policy} (k={cfg.wait_k}) max_write_per_chunk={cfg.max_write_per_chunk}", flush=True)
    print(f"[extrinsic] loading tokenizer {cfg.tokenizer_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    print(f"[extrinsic] loading model {cfg.model_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    for t in SPECIAL_TOKENS:
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
            )
            hyps.append(trace.hyp)
            # AL uses self-consistent y_len = len(tgt_word_g). Using
            # hyp.split() length here would mismatch g_list length whenever
            # BPE splits a word across a chunk boundary (~5-10% inflation),
            # yielding an artificially small AL.
            y_len_g = len(trace.tgt_word_g)
            al = compute_al(trace.tgt_word_g, trace.src_words, y_len_g)
            if al is not None:
                al_values.append(al)
            if trace.source_exhausted_without_eor:
                stream_stats["n_source_exhausted_without_eor"] += 1
            stream_stats["n_write_cap_hits"] += trace.write_cap_hits
            stream_stats["chunk_counts"].append(trace.chunks_committed)
            stream_stats.setdefault("per_sent", []).append({
                "idx": i, "src_words": trace.src_words,
                "y_len_g": y_len_g, "y_len_hyp": len(trace.hyp.split()),
                "chunks": trace.chunks_committed,
                "al": al, "g_words": trace.tgt_word_g,
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
        result["stream_stats"] = {
            "n_source_exhausted_without_eor": stream_stats["n_source_exhausted_without_eor"],
            "n_write_cap_hits": stream_stats["n_write_cap_hits"],
            "chunks_per_sent_mean": s.mean(stream_stats["chunk_counts"]),
            "chunks_per_sent_median": s.median(stream_stats["chunk_counts"]),
            "per_sent": stream_stats.get("per_sent", []),
        }
        print(f"[extrinsic] AL mean={result['al_mean']:.2f}  median={result['al_median']:.2f}  n_defined={result['al_n_defined']}/{len(pairs)}", flush=True)
        print(f"[extrinsic] chunks/sent mean={result['stream_stats']['chunks_per_sent_mean']:.2f}  median={result['stream_stats']['chunks_per_sent_median']:.1f}", flush=True)
        print(f"[extrinsic] source-exhausted-without-eor: {stream_stats['n_source_exhausted_without_eor']}/{len(pairs)}", flush=True)
        print(f"[extrinsic] write-cap hits: {stream_stats['n_write_cap_hits']}", flush=True)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[extrinsic] wrote {cfg.output}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="e.g. results/phase2/sft_condA_n10k/final")
    ap.add_argument("--tokenizer_dir",
                    default="/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended")
    ap.add_argument("--dev_src", required=True)
    ap.add_argument("--dev_ref", required=True)
    ap.add_argument("--latency", choices=list(LATENCY_TOKENS), default="medium")
    ap.add_argument("--n_sentences", type=int, default=-1)
    ap.add_argument("--max_new_tokens", type=int, default=200,
                    help="Offline mode: cap on decoded target tokens.")
    ap.add_argument("--mode", choices=["offline", "streaming"], default="offline")
    ap.add_argument("--policy", choices=["check_argmax", "wait_k"],
                    default="check_argmax",
                    help="Streaming policy. wait_k is the AL unit test.")
    ap.add_argument("--wait_k", type=int, default=3)
    ap.add_argument("--max_write_per_chunk", type=int, default=40,
                    help="Streaming: cap on target tokens per WRITE chunk before "
                         "forcing return to READ. If >5%% of chunks hit this, "
                         "the WRITE-stop mechanism is broken.")
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
    )
    run(cfg)


if __name__ == "__main__":
    main()
