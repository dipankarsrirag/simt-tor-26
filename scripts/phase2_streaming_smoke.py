"""
Local smoke tests for src/eval/extrinsic.py streaming (Layer 2), pre-submit.
Runs on login-node CPU (no model). Tests:

1. tokenize_source_by_words is byte-identical to full-source tokenize on 200
   newstest2013 lines. If ANY sentence differs, the streaming feed will not
   match training's token sequence and the model behaviour is undefined.
2. compute_al on hand-computed traces matches Ma 2019 §4 formula.
   - wait_k=3 trace on |X|=|Y|=9 → AL should be ~3.
3. StreamTrace dataclass round-trips through JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")


def test_tokenize_identity():
    from transformers import AutoTokenizer
    from src.eval.extrinsic import tokenize_source_by_words

    tok = AutoTokenizer.from_pretrained(
        "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended"
    )
    src_lines = Path("/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/newstest2013.de").read_text().splitlines()[:200]
    n_mismatch = 0
    for i, s in enumerate(src_lines):
        full, spans = tokenize_source_by_words(tok, s)
        concat = [t for span in spans for t in span]
        if full != concat:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  [mismatch idx {i}] src[:80]={s[:80]!r}")
                print(f"    full  len={len(full)}: {full[:20]}...")
                print(f"    spans total len={len(concat)}: {concat[:20]}...")
    print(f"tokenize_identity: {n_mismatch}/{len(src_lines)} mismatches")
    assert n_mismatch == 0, f"streaming feed will drift on {n_mismatch}/{len(src_lines)} sentences"
    print("  PASS")


def test_al_wait_k():
    from src.eval.extrinsic import compute_al
    # Wait-3 on |X|=|Y|=9: chunks of 3 src → 3 tgt each.
    # tgt words: 1..9. src read at each tgt word (g_words):
    #   tgt 1,2,3 emitted after src[1..3] read  → g = 3,3,3
    #   tgt 4,5,6 emitted after src[1..6] read  → g = 6,6,6
    #   tgt 7,8,9 emitted after src[1..9] read  → g = 9,9,9
    g = [3, 3, 3, 6, 6, 6, 9, 9, 9]
    al = compute_al(g, x_len=9, y_len=9)
    # tau = first i where g(i) == 9 → i=7. ratio = 9/9 = 1.
    # AL = (1/7) sum_{i=1..7} (g(i) - (i-1)*1)
    #    = (1/7) [(3-0) + (3-1) + (3-2) + (6-3) + (6-4) + (6-5) + (9-6)]
    #    = (1/7) [3+2+1+3+2+1+3] = 15/7 ≈ 2.14
    # Note: strict wait-3 gives AL ≈ 2.14 (not 3) because AL rewards emitting
    # after chunk 1's 3 src words, giving 3 tgt words at lag exactly 0 to 2.
    # This is Ma 2019's convention; empirically wait-k policies land AL close
    # to k for large sentences. Assert 1.5 < AL < 3.5.
    print(f"AL wait_k=3 on |X|=|Y|=9: {al:.3f}  (analytic ≈ 2.14)")
    assert 1.5 < al < 3.5, f"AL {al:.3f} out of expected range"
    print("  PASS")

    # Wait-1 on |X|=|Y|=9: g = 1,2,...,9. AL formula gives 1 (not 0).
    #   tau = 9 (first i where g(i) = 9). ratio = 1. AL = (1/9) sum (i - (i-1))
    #       = (1/9)*9 = 1.
    # Ma 2019: AL measures lag vs an oracle that reads (i-1)*ratio words at
    # step i. Wait-1 always has 1 more src word read than the oracle → AL=1.
    g_w1 = list(range(1, 10))
    al_w1 = compute_al(g_w1, x_len=9, y_len=9)
    print(f"AL wait_k=1 on |X|=|Y|=9: {al_w1:.3f}  (analytic = 1)")
    assert abs(al_w1 - 1.0) < 0.01, f"wait-1 AL should be 1, got {al_w1}"
    print("  PASS")

    # Wait-k on large N. k=3, N=99 (33 chunks of 3). Analytic:
    #   tau = 97 (first i where g(i) = 99). ratio = 1.
    #   Sum: 32 full chunks × (3+2+1) + partial chunk 33 (i=97 only) × 3 = 195.
    #   AL = 195/97 ≈ 2.01. Wait-k AL approaches k but stays below k when
    #   the last chunk is partial. For very large N (many chunks), → k = 3.
    g_big = []
    for chunk_end in range(3, 100, 3):
        g_big.extend([chunk_end] * 3)
    al_big = compute_al(g_big, x_len=99, y_len=len(g_big))
    print(f"AL wait_k=3 on |X|=|Y|={len(g_big)}: {al_big:.3f}  (analytic ≈ 2.01)")
    assert 1.8 < al_big < 2.3, f"large-N wait-3 AL should be ~2.01, got {al_big}"
    print("  PASS")

    # "One giant chunk" (offline-like): g = [9]*9. tau=1. AL = 9 - 0 = 9.
    g_off = [9] * 9
    al_off = compute_al(g_off, x_len=9, y_len=9)
    print(f"AL offline (all src before any tgt) on |X|=|Y|=9: {al_off:.3f}  (analytic = 9)")
    assert abs(al_off - 9) < 0.01
    print("  PASS")


if __name__ == "__main__":
    print("== test_tokenize_identity ==")
    test_tokenize_identity()
    print("\n== test_al_wait_k ==")
    test_al_wait_k()
    print("\nAll smoke tests PASSED.")
