"""A/B test: KV-cache-optimized annotator vs current naive full-forward version.

Verifies byte-identical divergence matrix output on 3 sample de-en sentences.
Then benchmarks speedup on 10 sentences.

Uses raw-mode prompt (no chat template) — the mode we validated as
tokenization-stable across prefixes.
"""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from transformers import AutoTokenizer, AutoConfig

from src.annotator.annotate import _prob_at_positions, _kv_cache_prefix_probs
from src.annotator.criterion import make_ot


MODEL = "/g/data/po67/dipankar/models/gemma-4-E2B"

SAMPLES = [
    ("Morales kriegerische Rhetorik habe jeden verbleibenden chilenischen Goodwill zerstört.",
     "Morales's belligerent rhetoric has sapped any residual Chilean goodwill."),
    ("Deutschland meldet einen richtungsweisenden Fall von West-Nil-Virus.",
     "Germany reports a groundbreaking case of West Nile virus."),
    ("Wenden Sie sich bitte an das Kundendienstzentrum, wenn Sie Fragen haben.",
     "Please contact the customer service center if you have any questions."),
]


def naive_full_matrix(model, tok, source, target, device):
    """Reference naive implementation — mirrors current annotate.py's inner loop."""
    src_ids = tok(source, add_special_tokens=False).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    n, m = len(src_ids), len(tgt_ids)
    SEP_id = tok("\n", add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    matrix = []
    p_full = None
    for i in range(1, n + 1):
        prompt_ids = bos + src_ids[:i] + SEP_id
        full_ids = prompt_ids + tgt_ids
        prefix_len = len(prompt_ids)
        positions = torch.arange(prefix_len - 1, prefix_len - 1 + m, device=device)
        inp = torch.tensor([full_ids], device=device, dtype=torch.long)
        p_i = _prob_at_positions(model, inp, positions)  # (m, V)
        matrix.append(p_i)
        if i == n:
            p_full = p_i
    return p_full, matrix


def kv_cache_matrix(model, tok, source, target, device):
    """KV-cache-optimized implementation."""
    src_ids = tok(source, add_special_tokens=False).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    SEP_id = tok("\n", add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    # NEW: prompt PREFIX is just [BOS], then extend with src tokens.
    # The SEP token is fed AS PART of the "final layer" per iteration.
    # Actually — the naive path puts SEP AFTER src[:i] and BEFORE target.
    # KV-cache version: src[:i] extended, then SEP + target on top per iteration.
    # But my _kv_cache_prefix_probs currently doesn't handle SEP separately.
    # Let me handle it inline: pre-append BOS, then treat "SEP + target" as
    # what gets fed after each src extension.

    # Actually the cleanest way: feed BOS once, extend by src token, then
    # for each iteration, add SEP to source-context first, then compute
    # target probs. But adding SEP would extend cache. To make byte-
    # identical: treat the growing prefix as BOS + src[:i] + SEP.

    # Simplest: use _kv_cache_prefix_probs where the "prompt prefix" is BOS + SEP
    # and the "src" being extended is different — that breaks the model. No.

    # Correct: prompt_prefix = BOS. src tokens = src_ids. But we need to
    # append SEP after last src token before target. So we treat it as
    # "src tokens plus a trailing SEP inserted per-iteration snapshot".

    # Easiest byte-identical version: extend with src[:i], then feed [SEP] + target
    # on the CLONED KV each iteration. That gives byte-identical probs since
    # the ATTENTION context is [BOS+src[:i]+SEP+target[:j]] for each j.

    # I'll inline this here rather than modify _kv_cache_prefix_probs.
    n = len(src_ids)
    m = len(tgt_ids)

    prompt_tensor = torch.tensor([bos], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=prompt_tensor, use_cache=True)
    kv = out.past_key_values

    matrix = []
    sep_plus_target_head = SEP_id + tgt_ids[:-1]  # (SEP + tgt[0..m-2]), fed after src[:i]
    sep_plus_target_head_tensor = torch.tensor([sep_plus_target_head], device=device, dtype=torch.long)

    p_full = None
    for i in range(1, n + 1):
        src_tok = torch.tensor([[src_ids[i - 1]]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=src_tok, past_key_values=kv, use_cache=True)
        kv = out.past_key_values  # KV = [BOS + src[:i]]

        from src.annotator.annotate import _clone_kv
        kv_snapshot = _clone_kv(kv)
        with torch.no_grad():
            out_sep_tgt = model(
                input_ids=sep_plus_target_head_tensor,
                past_key_values=kv_snapshot,
                use_cache=False,
            )
        # out_sep_tgt.logits shape (1, len(sep_plus_target_head), V) = (1, 1+m-1, V) = (1, m, V)
        # Position 0 in this forward = SEP token was fed; logit predicts P(next | ...+SEP) = P(target[0])
        # Position 1 = target[0] fed; logit predicts P(target[1] | ...)
        # ...
        # Position m-1 = target[m-2] fed; logit predicts P(target[m-1] | ...)
        # So all m target-position logits are in out_sep_tgt.logits[0].
        p_i = F.softmax(out_sep_tgt.logits[0].float(), dim=-1)  # (m, V)
        matrix.append(p_i)
        if i == n:
            p_full = p_i

    return p_full, matrix


def compare(matrix_a, matrix_b, tol=1e-4):
    """Return per-prefix max abs diff; also max across all."""
    diffs = []
    for a, b in zip(matrix_a, matrix_b):
        d = (a - b).abs().max().item()
        diffs.append(d)
    return max(diffs), diffs


def main():
    import torch.nn.functional as F
    global F  # for kv_cache_matrix
    print(f"Loading model {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    from transformers import Gemma3nForCausalLM
    if getattr(cfg, "model_type", None) == "gemma3n":
        model = Gemma3nForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    device = torch.device("cuda")
    model.to(device).eval()

    # A/B correctness on 3 sentences
    print("\n=== A/B correctness ===")
    for src, tgt in SAMPLES:
        t0 = time.time()
        p_full_a, m_a = naive_full_matrix(model, tok, src, tgt, device)
        t_a = time.time() - t0
        t0 = time.time()
        p_full_b, m_b = kv_cache_matrix(model, tok, src, tgt, device)
        t_b = time.time() - t0
        max_diff, _ = compare(m_a, m_b)
        n = len(m_a)
        print(f"  src({n} toks): {src[:50]!r}")
        print(f"    naive:    {t_a:.2f}s   kv-cache: {t_b:.2f}s   speedup {t_a/t_b:.2f}x")
        print(f"    max abs diff in probs: {max_diff:.6e}  (tol {1e-4:.0e})")
        p_full_diff = (p_full_a - p_full_b).abs().max().item()
        print(f"    p_full max diff: {p_full_diff:.6e}")

    # Benchmark on 10 sentences (repeat from samples)
    print("\n=== Benchmark (10 sentences) ===")
    bench = (SAMPLES * 4)[:10]
    t0 = time.time()
    for src, tgt in bench:
        _ = naive_full_matrix(model, tok, src, tgt, device)
    t_naive = time.time() - t0
    torch.cuda.synchronize()

    t0 = time.time()
    for src, tgt in bench:
        _ = kv_cache_matrix(model, tok, src, tgt, device)
    t_kv = time.time() - t0
    torch.cuda.synchronize()

    print(f"  naive:    {t_naive:.2f}s")
    print(f"  kv-cache: {t_kv:.2f}s")
    print(f"  speedup:  {t_naive/t_kv:.2f}x")


if __name__ == "__main__":
    main()
