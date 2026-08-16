"""
Tiny CPU sanity: annotate one 10-token sentence to catch shape/token
bugs before we spend GPU walltime. Runs in a few minutes on CPU.
"""

import sys
import time

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.annotator.annotate import annotate_pair
from src.constants import PRIMARY_BACKBONE


def main():
    print(f"Loading {PRIMARY_BACKBONE} on CPU ...")
    tok = AutoTokenizer.from_pretrained(str(PRIMARY_BACKBONE))
    model = AutoModelForCausalLM.from_pretrained(
        str(PRIMARY_BACKBONE), dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    src = "Die Katze schläft."
    tgt = "The cat is sleeping."
    n_src = len(tok(src, add_special_tokens=False)["input_ids"])
    n_tgt = len(tok(tgt, add_special_tokens=False)["input_ids"])
    print(f"source ({n_src} tok): {src!r}")
    print(f"target ({n_tgt} tok): {tgt!r}")

    t0 = time.time()
    ann = annotate_pair(
        model=model, tokenizer=tok, source=src, target=tgt,
        src_lang="German", tgt_lang="English", latency="medium",
        tau=0.05, criterion_name="js", verbose=True,
    )
    dt = time.time() - t0

    print(f"\nAnnotated in {dt:.1f}s")
    print(f"commit trace i*[j]: {ann.commit_source_tok_idx}")
    print(f"fired divergences: {[round(d, 4) if d != float('inf') else None for d in ann.fired_divergence]}")
    print(f"source_chunks ({len(ann.source_chunks)}): {ann.source_chunks}")
    print(f"target_chunks ({len(ann.target_chunks)}): {ann.target_chunks}")
    print(f"east_str: {ann.east_str}")

    # Structural checks that must hold regardless of divergence numbers.
    assert len(ann.commit_source_tok_idx) == n_tgt, "commit length must equal target token count"
    assert all(0 <= c <= n_src for c in ann.commit_source_tok_idx), "commit ids in [0, n]"
    for j in range(1, n_tgt):
        assert ann.commit_source_tok_idx[j] >= ann.commit_source_tok_idx[j - 1], "monotone"
    assert len(ann.source_chunks) == len(ann.target_chunks), "chunk counts equal"
    print("\nCPU TINY SMOKE OK")


if __name__ == "__main__":
    main()
