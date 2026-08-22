"""Probe: batched per-sentence prefix forward for the annotator.

Key idea: for a single (source, target) pair with source length n, currently
we do n sequential forward passes (one per prefix length i=1..n). Instead,
build a padded batch tensor of shape (n, L_max) where row i-1 is the input
for prefix length i, and process all n prefixes in ONE forward pass. GPU
parallelism amortizes the small-sequence overhead.

Byte-identical output check + benchmark vs sequential naive.
"""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig


MODEL = "/g/data/po67/dipankar/models/gemma-4-E2B"

SAMPLES = [
    ("Morales kriegerische Rhetorik habe jeden verbleibenden chilenischen Goodwill zerstört.",
     "Morales's belligerent rhetoric has sapped any residual Chilean goodwill."),
    ("Deutschland meldet einen richtungsweisenden Fall von West-Nil-Virus.",
     "Germany reports a groundbreaking case of West Nile virus."),
    ("Wenden Sie sich bitte an das Kundendienstzentrum, wenn Sie Fragen haben.",
     "Please contact the customer service center if you have any questions."),
]


def build_prompt_prefix_ids(tok):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return bos


def naive_sequential(model, tok, source, target, device):
    src_ids = tok(source, add_special_tokens=False).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    n, m = len(src_ids), len(tgt_ids)
    SEP = tok("\n", add_special_tokens=False).input_ids
    bos = build_prompt_prefix_ids(tok)

    matrix = []
    with torch.no_grad():
        for i in range(1, n + 1):
            row = bos + src_ids[:i] + SEP + tgt_ids
            prefix_len = len(bos) + i + len(SEP)
            inp = torch.tensor([row], device=device, dtype=torch.long)
            out = model(input_ids=inp, use_cache=False)
            positions = torch.arange(prefix_len - 1, prefix_len - 1 + m, device=device)
            p_i = F.softmax(out.logits[0, positions].float(), dim=-1)
            matrix.append(p_i)
    return matrix


def batched_all_prefixes(model, tok, source, target, device, batch_size=None):
    """Batched: build (n, L_max) padded tensor, one forward pass."""
    src_ids = tok(source, add_special_tokens=False).input_ids
    tgt_ids = tok(target, add_special_tokens=False).input_ids
    n, m = len(src_ids), len(tgt_ids)
    SEP = tok("\n", add_special_tokens=False).input_ids
    bos = build_prompt_prefix_ids(tok)
    pad_id = tok.pad_token_id or tok.eos_token_id or 0

    rows = []
    prefix_lens = []
    for i in range(1, n + 1):
        rows.append(bos + src_ids[:i] + SEP + tgt_ids)
        prefix_lens.append(len(bos) + i + len(SEP))
    max_len = max(len(r) for r in rows)

    matrix = [None] * n
    starts = list(range(0, n, batch_size or n))
    with torch.no_grad():
        for start in starts:
            end = min(start + (batch_size or n), n)
            batch_rows = rows[start:end]
            batch_prefix_lens = prefix_lens[start:end]
            B = end - start
            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            for k, r in enumerate(batch_rows):
                input_ids[k, :len(r)] = torch.tensor(r, device=device)
                attention_mask[k, :len(r)] = 1
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            # Extract per-row target logits
            for k in range(B):
                prefix_len = batch_prefix_lens[k]
                positions = torch.arange(prefix_len - 1, prefix_len - 1 + m, device=device)
                p_i = F.softmax(out.logits[k, positions].float(), dim=-1)
                matrix[start + k] = p_i
    return matrix


def compare(a, b):
    max_d = 0.0
    for i in range(len(a)):
        d = (a[i] - b[i]).abs().max().item()
        if d > max_d: max_d = d
    return max_d


def main():
    print(f"Loading model {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    if getattr(cfg, "model_type", None) == "gemma3n":
        from transformers import Gemma3nForCausalLM
        model = Gemma3nForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    device = torch.device("cuda")
    model.to(device).eval()

    print("\n=== A/B correctness (single sentences) ===")
    for src, tgt in SAMPLES:
        m_a = naive_sequential(model, tok, src, tgt, device)
        m_b = batched_all_prefixes(model, tok, src, tgt, device)
        max_d = compare(m_a, m_b)
        n = len(m_a)
        print(f"  src(n={n}): {src[:50]!r}   max_diff={max_d:.6e}")

    print("\n=== Benchmark (10 sentences repeated) ===")
    bench = (SAMPLES * 4)[:10]
    torch.cuda.synchronize()
    t0 = time.time()
    for src, tgt in bench:
        _ = naive_sequential(model, tok, src, tgt, device)
    torch.cuda.synchronize()
    t_naive = time.time() - t0

    t0 = time.time()
    for src, tgt in bench:
        _ = batched_all_prefixes(model, tok, src, tgt, device)
    torch.cuda.synchronize()
    t_batched = time.time() - t0

    print(f"  naive sequential: {t_naive:.2f}s")
    print(f"  batched (per-sent, all-prefixes): {t_batched:.2f}s")
    print(f"  speedup: {t_naive/t_batched:.2f}x")


if __name__ == "__main__":
    main()
