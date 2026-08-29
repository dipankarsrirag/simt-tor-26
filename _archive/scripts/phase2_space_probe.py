"""Space-separator probe: does inserting the standalone ▁ (id 236743) between
a source word and the argmax check unlock p(EOR)?

Hypothesis (2026-08-19 walkthrough): interleave() joins EAST-format parts with
' ', so the training string has a standalone '▁' between the last source-word
BPE and the '<|end-of-read|>' token. The model learns to predict EOR AFTER
the standalone '▁', not directly after the source-word's last BPE. At inference,
`stream_translate` feeds source words as `[▁word_bpes]` (leading space baked
into each word's tokenization) and never emits the standalone '▁' — so the
EOR prediction path is never queried.

This probe compares p(EOR | prefix_A) vs p(EOR | prefix_B) where
    A = [BOS, LAT, word_1_bpes, ..., word_i_bpes]
    B = A + [SPACE_ID]  # phantom standalone-space
at each word boundary of a handful of test sentences. If B >> A at chunk-
boundary positions AND B ~= A at mid-chunk positions, hypothesis confirmed.

Small test (20 sents on newstest2013, ~5 min on H200). No retraining, no state
change to any pipeline component.
"""

from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from src.annotator.east_format import END_OF_READ, LATENCY_TOKENS
from src.eval.extrinsic import tokenize_source_by_words

DEV_SRC = "/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/newstest2013.de"
CKPT = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/sft_n10k/final"
TOK_DIR = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended"
N_SENTS = 10
LATENCY = "medium"


@torch.no_grad()
def probe(model, tok, src: str, device: str, eor_id: int, space_id: int,
          lat_id: int, bos_id: int):
    """For each word-boundary position i in src, compute p(EOR) at:
        A = prefix ending in word_i's last BPE
        B = A + standalone SPACE
    Return list of (word_i, word_str, p_eor_A, p_eor_B, argmax_A, argmax_B).
    """
    words = src.split()
    # Use the SAME tokenize function as stream_translate — it has a fallback
    # path that handles first-word missing-leading-space via offset mapping.
    # This ensures the probe queries the exact same token positions inference does.
    full_ids, spans = tokenize_source_by_words(tok, src)

    # Feed prompt prefix
    ids_prefix = [bos_id, lat_id]
    ids_A = list(ids_prefix)
    results = []
    for wi, span in enumerate(spans):
        ids_A = ids_A + list(span)
        ids_B = ids_A + [space_id]
        # Forward-pass both (small; no need for cache).
        for tag, ids in [("A", ids_A), ("B", ids_B)]:
            t = torch.tensor([ids], device=device)
            out = model(input_ids=t, use_cache=False)
            probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
            p_eor = float(probs[eor_id].item())
            top1_id = int(probs.argmax().item())
            top1_p = float(probs[top1_id].item())
            top1_str = tok.convert_ids_to_tokens(top1_id)
            if tag == "A":
                p_eor_A, top1_A_id, top1_A_p, top1_A_str = p_eor, top1_id, top1_p, top1_str
            else:
                p_eor_B, top1_B_id, top1_B_p, top1_B_str = p_eor, top1_id, top1_p, top1_str
        results.append({
            "wi": wi, "word": words[wi],
            "p_eor_A": p_eor_A, "top1_A": top1_A_str, "top1_A_p": top1_A_p,
            "p_eor_B": p_eor_B, "top1_B": top1_B_str, "top1_B_p": top1_B_p,
            "argmax_A_is_eor": top1_A_id == eor_id,
            "argmax_B_is_eor": top1_B_id == eor_id,
        })
    return results


def main():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer {TOK_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    eor_id = tok(END_OF_READ, add_special_tokens=False).input_ids[0]
    space_id = tok(" ", add_special_tokens=False).input_ids[0]  # standalone ▁
    lat_id = tok(LATENCY_TOKENS[LATENCY], add_special_tokens=False).input_ids[0]
    bos_id = tok.bos_token_id
    print(f"  eor_id = {eor_id}  space_id = {space_id}  lat_id = {lat_id}  bos_id = {bos_id}", flush=True)
    print(f"  standalone SPACE tokenizes to: {tok.convert_ids_to_tokens(space_id)!r}", flush=True)

    print(f"Loading model {CKPT}", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(CKPT)
    if getattr(cfg, "model_type", None) == "gemma3n":
        from transformers import Gemma3nForCausalLM
        model = Gemma3nForCausalLM.from_pretrained(CKPT, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    with open(DEV_SRC) as f:
        srcs = [l.strip() for l in f if l.strip()][:N_SENTS]
    print(f"Probing {len(srcs)} sentences...\n", flush=True)

    # Aggregate stats
    n_A_eor_argmax = 0
    n_B_eor_argmax = 0
    n_total_positions = 0
    p_eor_A_all = []
    p_eor_B_all = []

    for i, src in enumerate(srcs):
        results = probe(model, tok, src, device, eor_id, space_id, lat_id, bos_id)
        if results is None:
            print(f"  [{i}] SKIP (naive tokenize mismatch)")
            continue
        print(f"  [{i}] SRC: {src[:80]!r}", flush=True)
        header = f"    {'wi':>3} {'word':<15s} {'p(EOR|A)':>10} {'top1(A)':<25s} {'p(EOR|B)':>10} {'top1(B)':<25s}  {'⇑?':>3}"
        print(header, flush=True)
        for r in results:
            arrow = ""
            if r["argmax_A_is_eor"]: arrow += "A→EOR "
            if r["argmax_B_is_eor"]: arrow += "B→EOR"
            print(f"    {r['wi']:>3} {r['word'][:15]:<15s} {r['p_eor_A']:>10.4f} {r['top1_A'][:25]:<25s} {r['p_eor_B']:>10.4f} {r['top1_B'][:25]:<25s}  {arrow}",
                  flush=True)
            p_eor_A_all.append(r['p_eor_A'])
            p_eor_B_all.append(r['p_eor_B'])
            if r['argmax_A_is_eor']: n_A_eor_argmax += 1
            if r['argmax_B_is_eor']: n_B_eor_argmax += 1
            n_total_positions += 1
        print()

    import statistics as s
    print(f"\n=== AGGREGATE OVER {n_total_positions} word-boundary positions ({len(srcs)} sents) ===")
    print(f"  Prompt A (last-BPE-of-word):  mean p(EOR) = {s.mean(p_eor_A_all):.6f}  argmax=EOR at {n_A_eor_argmax}/{n_total_positions} positions")
    print(f"  Prompt B (+standalone ▁):     mean p(EOR) = {s.mean(p_eor_B_all):.6f}  argmax=EOR at {n_B_eor_argmax}/{n_total_positions} positions")
    ratio = s.mean(p_eor_B_all) / max(s.mean(p_eor_A_all), 1e-10)
    print(f"  p(EOR|B) / p(EOR|A) mean ratio = {ratio:.2f}x")
    print(f"\nHypothesis confirmed iff: p(EOR|B) >> p(EOR|A) AND B→EOR count >> A→EOR count")

if __name__ == "__main__":
    main()
