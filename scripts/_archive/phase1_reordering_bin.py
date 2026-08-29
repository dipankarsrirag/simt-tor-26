"""
Gate-1 analysis: stratified-by-reordering aggregate.

Reads a phase1_tau_sweep matrices.jsonl plus the precomputed GPT-4 per-sentence
Pearson (from phase1_precompute_gpt4_pearson.py). Bins sentences by GPT-4's
own per-sentence Pearson as a reordering-severity proxy — low Pearson = more
reordering. Reports per bin:

  * mean chunk-count delta vs GPT-4 (|ours - gpt4|)
  * per-sentence Pearson (ours) at matched-chunk-count tau
  * MATCH rate — fraction of sentences where ours' Pearson < 0.85 (i.e., we
    also read the sentence as non-monotonic)
  * mean matched tau, mean ours_chunks

Bin thresholds are read from the gate1_indices.json file (fixed absolute per
2026-08-16 DECISION); never quintile-based.

Usage:
    python scripts/phase1_reordering_bin.py \\
        --matrices results/phase1_tau_sweep_ot_n200/matrices.jsonl \\
        --gpt4_pearson_full results/gate1/gpt4_pearson_full.json \\
        --output results/gate1/reordering_bin_ot_n200.json

The script does not need the tokenizer — it recomputes ours' commit trace
directly from the (n, m) divergence matrix at each candidate tau.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone
from src.constants import REPO_ROOT


MATCH_THRESHOLD = 0.85  # ours_pearson < this counts as MATCH


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
    if not matrix or not matrix[0]:
        return []
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
    groups, j = 0, 0
    while j < m:
        k = j
        while k + 1 < m and commit[k + 1] == commit[j]:
            k += 1
        groups += 1
        j = k + 1
    return groups


def matched_count_commit(matrix, n, target_chunk_count, tau_grid):
    """For a sentence, pick the tau whose ours-chunk-count is closest to
    target_chunk_count. Returns (best_tau, best_commit, best_chunks, best_delta)."""
    best = (None, None, None, float("inf"))
    for tau in tau_grid:
        c = commit_from_matrix(matrix, tau, n)
        kk = chunk_count(c)
        delta = abs(kk - target_chunk_count)
        if delta < best[3]:
            best = (tau, c, kk, delta)
            if delta == 0:
                break
    return best


def summarise(vals):
    finite = [v for v in vals if v is not None and not math.isnan(v)]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    finite_sorted = sorted(finite)
    return {
        "n": len(finite),
        "mean": sum(finite) / len(finite),
        "median": finite_sorted[len(finite) // 2],
        "min": finite_sorted[0],
        "max": finite_sorted[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", required=True,
                    help="Path to phase1_tau_sweep matrices.jsonl.")
    ap.add_argument("--gpt4_pearson_full", required=True,
                    help="Path to results/gate1/gpt4_pearson_full.json (from "
                         "phase1_precompute_gpt4_pearson.py).")
    ap.add_argument("--tau_grid",
                    default="0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00",
                    help="Tau candidates to search over for per-sentence "
                         "matched-count tau. Default fits OT range; JS uses "
                         "e.g. 0.02,0.05,0.10,0.15,0.20,0.30.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tau_grid = [float(x) for x in args.tau_grid.split(",")]

    gpt4_full = json.loads(Path(args.gpt4_pearson_full).read_text())
    gpt4_per_sent = gpt4_full["per_sentence"]  # keys are string indices
    # Normalise keys to int for lookup by rec["index"].
    gpt4_per_sent = {int(k): v for k, v in gpt4_per_sent.items()}
    thresholds = gpt4_full["config"]
    print(f"GPT-4 Pearson bins: mono ≥ {thresholds['mono_thresh']}, "
          f"reord < {thresholds['reord_thresh']}")

    per_sentence = []
    with open(args.matrices) as f:
        for line in f:
            per_sentence.append(json.loads(line))
    print(f"Loaded {len(per_sentence)} matrices from {args.matrices}")

    by_bin = defaultdict(list)
    missing = 0
    for rec in per_sentence:
        idx = rec["index"]
        if idx not in gpt4_per_sent:
            missing += 1
            continue
        gpt4_info = gpt4_per_sent[idx]
        b = gpt4_info["bin"]
        if b == "undefined":
            continue

        n = rec["n_src_tok"]
        m = rec["n_tgt_tok"]
        gpt4_chunks = gpt4_info["gpt4_chunks"]
        gpt4_pearson = gpt4_info["pearson"]

        tau, commit, ours_chunks, delta = matched_count_commit(
            rec["matrix"], n, gpt4_chunks, tau_grid
        )
        if commit is None or m == 0:
            continue
        xs = [c / n for c in commit]
        ys = [(j + 1) / m for j in range(m)]
        ours_pearson = pearson(xs, ys)

        # match=None means the trace is degenerate (single-chunk collapse or
        # NaN Pearson) — these are single-chunk-with-FP-noise cases which
        # can't meaningfully agree with GPT-4 on non-monotonicity. Excluding
        # them keeps the metric honest.
        if (ours_pearson is None or math.isnan(ours_pearson)
                or ours_chunks is None or ours_chunks <= 1):
            match = None
        else:
            match = ours_pearson < MATCH_THRESHOLD

        by_bin[b].append({
            "index": idx,
            "latency": rec["latency"],
            "gpt4_pearson": gpt4_pearson,
            "gpt4_chunks": gpt4_chunks,
            "ours_pearson": ours_pearson,
            "ours_chunks": ours_chunks,
            "chunk_delta": delta,
            "matched_tau": tau,
            "n_src_tok": n,
            "n_tgt_tok": m,
            "match": match,
        })

    if missing:
        print(f"  WARNING: {missing} sentences in matrices had no entry in gpt4_pearson_full.json")

    print(f"\nSentences per bin (after matching to gpt4 pearson full):")
    for b in ["monotone", "mild", "reordering"]:
        print(f"  {b:>12s}: {len(by_bin[b])}")

    # Aggregate per bin.
    report = {"config": {
        "matrices": args.matrices,
        "gpt4_pearson_full": args.gpt4_pearson_full,
        "tau_grid": tau_grid,
        "match_threshold": MATCH_THRESHOLD,
        "bin_thresholds": {"monotone": thresholds["mono_thresh"],
                           "reordering": thresholds["reord_thresh"]},
    }, "per_bin": {}}

    print(f"\n=== Per-bin aggregates ===")
    print(f"Coverage = fraction of bin where matched-count trace has >1 chunk (Pearson defined).")
    print(f"MATCH%_cond = fraction of *covered* sentences where ours_pearson < {MATCH_THRESHOLD}.")
    print(f"MATCH%_eff  = fraction of *all* bin sentences where covered AND ours_pearson < {MATCH_THRESHOLD}")
    print(f"              (treats single-chunk collapse as MISS — the honest number for the mechanism claim).")
    print()
    print(f"{'bin':>12s}  {'n':>4s}  {'cov%':>5s}  {'gpt4_ch':>7s}  {'ours_ch':>7s}  "
          f"{'delta':>6s}  {'gpt4_p':>7s}  {'ours_p':>7s}  "
          f"{'MATCH%_cond':>11s}  {'MATCH%_eff':>10s}  {'tau_med':>8s}")

    for b in ["monotone", "mild", "reordering"]:
        rows = by_bin[b]
        if not rows:
            continue
        # Filter out NaN Pearson for the "ours_pearson" summary too.
        agg = {
            "n": len(rows),
            "gpt4_chunks": summarise([r["gpt4_chunks"] for r in rows]),
            "ours_chunks": summarise([r["ours_chunks"] for r in rows]),
            "chunk_delta": summarise([r["chunk_delta"] for r in rows]),
            "gpt4_pearson": summarise([r["gpt4_pearson"] for r in rows]),
            "ours_pearson": summarise([r["ours_pearson"] for r in rows]),
            "matched_tau": summarise([r["matched_tau"] for r in rows]),
        }
        covered = [r for r in rows if r["match"] is not None]
        agg["covered_n"] = len(covered)
        agg["coverage"] = len(covered) / len(rows) if rows else None
        agg["single_chunk_n"] = sum(1 for r in rows if r["ours_chunks"] == 1)
        match_cond_count = sum(1 for r in covered if r["match"])
        agg["match_n"] = len(covered)
        agg["match_rate_conditional"] = (
            match_cond_count / len(covered) if covered else None
        )
        agg["match_rate_effective"] = match_cond_count / len(rows) if rows else None

        report["per_bin"][b] = {"aggregate": agg, "sentences": rows}

        cov = f"{agg['coverage']*100:4.1f}%"
        mrc = f"{agg['match_rate_conditional']*100:5.1f}%" if agg["match_rate_conditional"] is not None else "   --"
        mre = f"{agg['match_rate_effective']*100:5.1f}%" if agg["match_rate_effective"] is not None else "   --"
        print(f"{b:>12s}  {agg['n']:>4d}  {cov:>5s}  "
              f"{agg['gpt4_chunks']['mean']:>7.2f}  "
              f"{agg['ours_chunks']['mean']:>7.2f}  "
              f"{agg['chunk_delta']['mean']:>6.2f}  "
              f"{agg['gpt4_pearson']['mean']:>7.3f}  "
              f"{agg['ours_pearson']['mean']:>7.3f}  "
              f"{mrc:>11s}  {mre:>10s}  "
              f"{agg['matched_tau']['median']:>8.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
