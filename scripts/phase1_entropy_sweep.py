"""
Entropy-only ablation sweep from a matrices.jsonl produced by
phase1_tau_sweep.py with --record_entropy.

For each threshold H_tau in the grid:
  * commit criterion: first i where H(P_pre[i][j]) < H_tau
  * fallback: i = n if criterion never fires
  * monotonicity enforced
  * chunk grouping and Pearson(i*/n, j/m) as in the JS sweep

The entropy-only baseline answers METHOD §3's question: is the full-source
oracle doing any work, or is "when does the model become confident" alone
enough to place chunks?

If entropy-only matches JS (both diagonal, similar chunk counts, similar
Pearson), P_full is not adding information — the criterion is dominated
by prefix-length confidence growth. Conversely, if JS has a lower
Pearson_median at similar fire rates, the oracle IS doing work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone
from src.constants import REPO_ROOT


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


def commit_from_entropy(entropy_matrix, h_tau, n):
    """First i where H(P_pre[i][j]) < h_tau, else n. Monotone-enforced."""
    m = len(entropy_matrix[0])
    commit = [n] * m
    for i_row, row in enumerate(entropy_matrix):
        i = i_row + 1
        for j in range(m):
            if commit[j] < n:
                continue
            h = row[j]
            if not math.isnan(h) and h < h_tau:
                commit[j] = i
    return _enforce_monotone(commit)


def chunk_count(commit):
    m = len(commit)
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
    ap.add_argument("--taus", default="0.5,1.0,2.0,3.0,4.0,5.0,6.0",
                    help="Entropy thresholds in nats. Gemma-4 vocab=262144 → "
                         "max entropy ~12.5 nats.")
    ap.add_argument("--output")
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]

    per_sentence = []
    with open(args.matrices) as f:
        for line in f:
            rec = json.loads(line)
            if "entropy_matrix" not in rec:
                sys.exit(f"matrices file does not include 'entropy_matrix' — re-run tau sweep with --record_entropy")
            per_sentence.append(rec)
    print(f"Loaded {len(per_sentence)} sentences from {args.matrices}")

    out = []
    for h_tau in taus:
        fires = []
        commits_frac = []
        chunks_ours = []
        chunks_gpt4 = []
        pearsons = []
        for rec in per_sentence:
            n = rec["n_src_tok"]
            m = rec["n_tgt_tok"]
            commit = commit_from_entropy(rec["entropy_matrix"], h_tau, n)
            fired = [c for c in commit if c < n]
            fires.append(len(fired) > 0)
            commits_frac.append(len(fired) / m if m > 0 else 0.0)
            chunks_ours.append(chunk_count(commit))
            chunks_gpt4.append(len(rec["gpt4_source_chunks"]))
            xs = [c / n for c in commit]
            ys = [(j + 1) / m for j in range(m)]
            pearsons.append(pearson(xs, ys))

        row = {
            "h_tau": h_tau,
            "fire_fraction": sum(fires) / max(len(fires), 1),
            "mean_committed_fraction": sum(commits_frac) / max(len(commits_frac), 1),
            "chunks_per_sentence_mean_ours": sum(chunks_ours) / max(len(chunks_ours), 1),
            "chunks_per_sentence_mean_gpt4": sum(chunks_gpt4) / max(len(chunks_gpt4), 1),
            "pearson_i_over_n_vs_j_over_m": summarise(pearsons),
        }
        out.append(row)

    result = {"config": vars(args), "rows": out}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.output}\n")

    print(f"{'h_tau':>6}  {'fire%':>6}  {'commit%':>8}  {'ours_ch':>7}  {'gpt4_ch':>7}  "
          f"{'pear_med':>8}  {'pear_min':>8}  {'#NaN':>4}")
    for row in out:
        p = row["pearson_i_over_n_vs_j_over_m"]
        def f(x): return f"{x:.3f}" if x is not None else "  --"
        print(f"{row['h_tau']:>6.2f}  "
              f"{row['fire_fraction']*100:>5.1f}%  "
              f"{row['mean_committed_fraction']*100:>7.1f}%  "
              f"{row['chunks_per_sentence_mean_ours']:>7.2f}  "
              f"{row['chunks_per_sentence_mean_gpt4']:>7.2f}  "
              f"{f(p['median']):>8}  {f(p['min']):>8}  {p['n_nan']:>4}")


if __name__ == "__main__":
    main()
