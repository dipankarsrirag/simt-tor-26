"""
Per-sentence comparison: does OUR criterion track the same
monotonicity/reordering structure as GPT-4 on the same sentence?

For each sentence, pick the tau_js at which OUR chunk count is closest
to GPT-4's chunk count for that sentence. Compute:
  * per-sentence Pearson under GPT-4 chunks (as in phase1_gpt4_pearson)
  * per-sentence Pearson under OUR chunks at the matched-count tau
  * Pearson-of-Pearsons across sentences

If per-sentence Pearsons correlate strongly (r > 0.7 across the 48
sentences), our criterion IS tracking the same structure GPT-4 does,
including reordering. If uncorrelated, we're missing GPT-4's signal —
and OT / Qwen become worth trying.

Also reports the top-K reordering candidates (lowest GPT-4 Pearson)
with side-by-side commit traces from GPT-4 and ours, so we can eyeball
whether our criterion catches the German verb-final / fronted-subject
cases that CLAUDE.md predicts should distinguish us.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone
from src.constants import PRIMARY_BACKBONE


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return float("nan")
    return sxy / denom


def commit_from_matrix(matrix, tau, n):
    m = len(matrix[0])
    commit = [n] * m
    for i_row, row in enumerate(matrix):
        i = i_row + 1
        for j in range(m):
            if commit[j] < n:
                continue
            d = row[j]
            if not math.isnan(d) and d < tau:
                commit[j] = i
    return _enforce_monotone(commit)


def chunk_count(commit):
    m = len(commit)
    if m == 0:
        return 0
    groups = 0
    j = 0
    while j < m:
        k = j
        while k + 1 < m and commit[k + 1] == commit[j]:
            k += 1
        groups += 1
        j = k + 1
    return groups


def gpt4_commit_trace(gpt4_src_chunks, gpt4_tgt_chunks, tokenizer):
    K = len(gpt4_src_chunks)
    tok_no_special = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    cum_src, cum_tgt = [], []
    for k in range(K):
        cum_src.append(len(tok_no_special(" ".join(gpt4_src_chunks[: k + 1]))))
        cum_tgt.append(len(tok_no_special(" ".join(gpt4_tgt_chunks[: k + 1]))))
    total_src = cum_src[-1]
    total_tgt = cum_tgt[-1]
    commit = []
    prev = 0
    for k in range(K):
        for _ in range(prev, cum_tgt[k]):
            commit.append(cum_src[k])
        prev = cum_tgt[k]
    return _enforce_monotone(commit), total_src, total_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--tokenizer_path", default=str(PRIMARY_BACKBONE))
    ap.add_argument("--tau_grid",
                    default="0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30",
                    help="Fine grid for tau_js — used to find per-sentence "
                         "matched-count tau.")
    ap.add_argument("--top_k_reorder", type=int, default=5)
    ap.add_argument("--output")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    taus = [float(x) for x in args.tau_grid.split(",")]

    per_sentence = []
    with open(args.matrices) as f:
        for line in f:
            per_sentence.append(json.loads(line))
    print(f"Loaded {len(per_sentence)} sentences from {args.matrices}")

    rows = []
    for rec in per_sentence:
        n = rec["n_src_tok"]
        m = rec["n_tgt_tok"]

        # GPT-4 side.
        gpt4_commit, gpt4_n, gpt4_m = gpt4_commit_trace(
            rec["gpt4_source_chunks"], rec["gpt4_target_chunks"], tokenizer
        )
        gpt4_k = len(rec["gpt4_source_chunks"])
        gpt4_xs = [c / gpt4_n for c in gpt4_commit]
        gpt4_ys = [(j + 1) / len(gpt4_commit) for j in range(len(gpt4_commit))]
        gpt4_p = pearson(gpt4_xs, gpt4_ys)

        # Ours: pick per-sentence tau_js whose chunk count is closest to gpt4_k.
        best_tau, best_commit, best_delta = None, None, float("inf")
        for tau in taus:
            c = commit_from_matrix(rec["matrix"], tau, n)
            kk = chunk_count(c)
            delta = abs(kk - gpt4_k)
            if delta < best_delta:
                best_delta = delta
                best_tau = tau
                best_commit = c
                if delta == 0:
                    break
        ours_xs = [c / n for c in best_commit]
        ours_ys = [(j + 1) / m for j in range(m)]
        ours_p = pearson(ours_xs, ours_ys)
        ours_k = chunk_count(best_commit)

        rows.append({
            "index": rec["index"],
            "latency": rec["latency"],
            "n_src_our": n, "n_tgt_our": m,
            "n_src_gpt4": gpt4_n, "n_tgt_gpt4": gpt4_m,
            "gpt4_chunks": gpt4_k,
            "ours_chunks_at_matched": ours_k,
            "chunk_count_delta": best_delta,
            "gpt4_pearson": gpt4_p,
            "ours_pearson": ours_p,
            "ours_tau_js": best_tau,
            "gpt4_commit": gpt4_commit,
            "ours_commit": best_commit,
            "gpt4_source_chunks": rec["gpt4_source_chunks"],
            "gpt4_target_chunks": rec["gpt4_target_chunks"],
        })

    # Aggregates.
    gpt4_ps = [r["gpt4_pearson"] for r in rows if not math.isnan(r["gpt4_pearson"])]
    ours_ps = [r["ours_pearson"] for r in rows if not math.isnan(r["ours_pearson"])]
    # Pearson-of-Pearsons across matched sentences (both finite).
    paired = [(r["gpt4_pearson"], r["ours_pearson"]) for r in rows
              if not math.isnan(r["gpt4_pearson"]) and not math.isnan(r["ours_pearson"])]
    p_of_p = pearson([a for a, _ in paired], [b for _, b in paired]) if len(paired) >= 2 else float("nan")

    def med(xs): return sorted(xs)[len(xs) // 2] if xs else None

    print(f"\n=== Aggregate ===")
    print(f"GPT-4  chunks/sent: mean={sum(r['gpt4_chunks'] for r in rows)/len(rows):.2f}")
    print(f"Ours   chunks/sent: mean={sum(r['ours_chunks_at_matched'] for r in rows)/len(rows):.2f}")
    print(f"Chunk-count delta: mean_abs={sum(r['chunk_count_delta'] for r in rows)/len(rows):.2f}")
    print(f"GPT-4  Pearson: min={min(gpt4_ps):.3f} median={med(gpt4_ps):.3f} max={max(gpt4_ps):.3f}")
    print(f"Ours   Pearson (matched-count tau): min={min(ours_ps):.3f} median={med(ours_ps):.3f} max={max(ours_ps):.3f}")
    print(f"Per-sentence Pearson-of-Pearsons (GPT-4 vs Ours): r={p_of_p:.3f}  n={len(paired)}")

    # Reordering candidates — the K lowest GPT-4 Pearson sentences.
    reord = sorted([r for r in rows if not math.isnan(r["gpt4_pearson"])],
                   key=lambda r: r["gpt4_pearson"])[: args.top_k_reorder]
    print(f"\n=== Top {args.top_k_reorder} reordering candidates (lowest GPT-4 Pearson) ===")
    for r in reord:
        agree = "MATCH" if r["ours_pearson"] < 0.85 else "MISS"
        print(f"idx={r['index']:>6d} lat={r['latency']:>6s}  "
              f"gpt4={r['gpt4_pearson']:.3f}  ours@k={r['ours_chunks_at_matched']:>2d}={r['ours_pearson']:.3f}  "
              f"[{agree}]")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "aggregate": {
                "gpt4_pearson_median": med(gpt4_ps),
                "gpt4_pearson_min": min(gpt4_ps) if gpt4_ps else None,
                "ours_pearson_median": med(ours_ps),
                "ours_pearson_min": min(ours_ps) if ours_ps else None,
                "per_sentence_pearson_of_pearsons": p_of_p,
                "n_paired": len(paired),
                "gpt4_chunks_mean": sum(r["gpt4_chunks"] for r in rows) / len(rows),
                "ours_chunks_mean": sum(r["ours_chunks_at_matched"] for r in rows) / len(rows),
                "chunk_delta_mean_abs": sum(r["chunk_count_delta"] for r in rows) / len(rows),
            },
            "per_sentence": rows,
        }, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
