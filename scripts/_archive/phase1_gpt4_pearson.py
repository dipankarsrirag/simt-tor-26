"""
Compute GPT-4's Pearson(i*/n, j/m) on the same 48 sentences we ran ours on.

Discriminates:
  * If GPT-4 Pearson_med ~ 0.9: WMT De->En at 30-50 tokens is largely
    monotonic; "our criterion is diagonal" isn't degeneracy, it's the
    ground truth of the data (EAST App. C already selects monotonic
    pairs). Ours matching GPT-4 to within noise becomes a positive
    result; RWTH is then the arbiter of tag quality.
  * If GPT-4 Pearson_med << ours: GPT-4 finds reordering our criterion
    doesn't. Then Qwen + OT are worth trying.

Uses gpt4_source_chunks / gpt4_target_chunks fields already in
matrices.jsonl. Also prints the 3 lowest-Pearson sentences (both GPT-4
and ours at a matched chunk-count tau) so we can eyeball reordering.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone
from src.constants import PRIMARY_BACKBONE, REPO_ROOT


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


def gpt4_commit_trace(gpt4_src_chunks, gpt4_tgt_chunks, tokenizer):
    """Convert GPT-4's chunk pairs into a per-target-token commit trace.

    For chunk k with target span [j_start_k, j_end_k) and cumulative
    source-token count s_k = sum of tok-lengths of src_chunks[0..k+1]:
        commit[j] = s_k for j in [j_start_k, j_end_k)

    Uses cumulative whitespace-join tokenisation (which matches how the
    full source tokenises on our data — verified in Phase 0 join-equiv).
    """
    K = len(gpt4_src_chunks)
    assert K == len(gpt4_tgt_chunks), "chunk count mismatch"

    tok_no_special = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]

    # Cumulative source and target token lengths at each chunk boundary.
    cum_src = []
    cum_tgt = []
    for k in range(K):
        src_prefix_str = " ".join(gpt4_src_chunks[: k + 1])
        tgt_prefix_str = " ".join(gpt4_tgt_chunks[: k + 1])
        cum_src.append(len(tok_no_special(src_prefix_str)))
        cum_tgt.append(len(tok_no_special(tgt_prefix_str)))

    # Total target tokens and source tokens (from the FULL sentence).
    total_src = cum_src[-1]
    total_tgt = cum_tgt[-1]

    commit = []
    prev_tgt_end = 0
    for k in range(K):
        tgt_end = cum_tgt[k]
        for _ in range(prev_tgt_end, tgt_end):
            commit.append(cum_src[k])
        prev_tgt_end = tgt_end

    # Enforce monotonicity (already true by construction here).
    commit = _enforce_monotone(commit)
    return commit, total_src, total_tgt


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


def summarise(vals):
    finite = sorted(v for v in vals if not math.isnan(v))
    if not finite:
        return {"n_defined": 0, "n_nan": len(vals),
                "min": None, "median": None, "max": None}
    return {"n_defined": len(finite), "n_nan": len(vals) - len(finite),
            "min": finite[0], "median": finite[len(finite) // 2],
            "max": finite[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--tokenizer_path", default=str(PRIMARY_BACKBONE))
    ap.add_argument("--output")
    ap.add_argument("--top_lowest", type=int, default=5,
                    help="Print details of the N sentences with lowest "
                         "GPT-4 Pearson (reordering candidates).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    per_sentence = []
    with open(args.matrices) as f:
        for line in f:
            per_sentence.append(json.loads(line))
    print(f"Loaded {len(per_sentence)} sentences from {args.matrices}")

    rows = []
    for rec in per_sentence:
        try:
            commit, cum_src_total, cum_tgt_total = gpt4_commit_trace(
                rec["gpt4_source_chunks"], rec["gpt4_target_chunks"], tokenizer
            )
        except Exception as e:
            print(f"  idx={rec['index']} FAILED: {type(e).__name__}: {e}")
            continue
        # Use GPT-4's own token counts (from its chunks) for the axes —
        # the annotator's n/m are from our tokenisation of the raw
        # source/target which may differ by 1-2 subwords; keep this
        # comparison internally consistent to GPT-4's cumulative counts.
        m = len(commit)
        if cum_src_total == 0 or m == 0:
            continue
        xs = [c / cum_src_total for c in commit]
        ys = [(j + 1) / m for j in range(m)]
        r = pearson(xs, ys)
        rows.append({
            "index": rec["index"],
            "latency": rec["latency"],
            "n_our_src": rec["n_src_tok"],
            "n_our_tgt": rec["n_tgt_tok"],
            "n_gpt4_src": cum_src_total,
            "n_gpt4_tgt": cum_tgt_total,
            "gpt4_chunks": len(rec["gpt4_source_chunks"]),
            "gpt4_commit": commit,
            "gpt4_source_chunks": rec["gpt4_source_chunks"],
            "gpt4_target_chunks": rec["gpt4_target_chunks"],
            "pearson": r,
        })

    print(f"Computed for {len(rows)} sentences")

    pearsons = [r["pearson"] for r in rows]
    chunks = [r["gpt4_chunks"] for r in rows]
    summary = {
        "n_sentences": len(rows),
        "gpt4_chunks_mean": sum(chunks) / max(len(chunks), 1),
        "pearson_i_over_n_vs_j_over_m": summarise(pearsons),
    }
    if args.output:
        Path(args.output).write_text(json.dumps({"summary": summary, "per_sentence": rows}, indent=2, ensure_ascii=False))
        print(f"Wrote {args.output}\n")

    p = summary["pearson_i_over_n_vs_j_over_m"]
    print(f"GPT-4 Pearson(i*/n, j/m):")
    print(f"  min={p['min']:.3f} median={p['median']:.3f} max={p['max']:.3f}")
    print(f"  #defined={p['n_defined']} #NaN={p['n_nan']}")
    print(f"  mean chunks/sentence: {summary['gpt4_chunks_mean']:.2f}")

    # Lowest-Pearson sentences — the reordering candidates.
    finite_rows = [r for r in rows if not math.isnan(r["pearson"])]
    finite_rows.sort(key=lambda r: r["pearson"])
    print(f"\n=== {args.top_lowest} sentences with LOWEST GPT-4 Pearson (reordering candidates) ===")
    for r in finite_rows[:args.top_lowest]:
        print(f"\nidx={r['index']} lat={r['latency']} pearson={r['pearson']:.3f} "
              f"n_src={r['n_gpt4_src']} n_tgt={r['n_gpt4_tgt']} chunks={r['gpt4_chunks']}")
        for k, (src_c, tgt_c) in enumerate(zip(r["gpt4_source_chunks"], r["gpt4_target_chunks"])):
            print(f"  chunk {k}: SRC={src_c!r}")
            print(f"           TGT={tgt_c!r}")


if __name__ == "__main__":
    main()
