"""
Random-at-matched-latency floor from results/phase1_tau_sweep/matrices.jsonl.

For each tau in the previous sweep:
  1. For each sentence, count `k` = number of consecutive-run groups in
     the JS-derived commit trace at that tau (== #chunks).
  2. Draw `k` distinct source-token indices uniformly from 1..n, sort,
     enforce monotonicity, expand into a per-target-token commit trace
     that matches the JS trace's chunk-count and latency by construction.
  3. Compute Pearson(i*/n, j/m) on the random trace.
  4. Report the median / min / max of the random-trace Pearson
     distribution across sentences, alongside the JS numbers.

If JS Pearson_median is not below the random Pearson_median at any tau,
JS has no signal on Gemma-4-E2B under the raw-concat prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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


def commit_from_matrix(matrix, tau, n):
    m = len(matrix[0])
    commit = [n] * m
    for i_row, row in enumerate(matrix):
        i = i_row + 1
        for j in range(m):
            d = row[j]
            if commit[j] < n:
                continue
            if not math.isnan(d) and d < tau:
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


def random_commit(n: int, m: int, k: int, rng: random.Random) -> list[int]:
    """Random monotone commit trace with exactly `k` chunks over m target tokens.

    Sample k distinct source-token boundaries uniformly from 1..n, sort.
    Distribute the m target tokens across k chunks by sampling k-1 split
    positions from 1..m-1. Each chunk of consecutive target tokens shares
    the corresponding boundary.
    """
    if k <= 0:
        return [n] * m
    if k >= m:
        # More chunks than tokens: one chunk per token (up to m).
        src_bounds = sorted(rng.sample(range(1, n + 1), min(k, n)))
        # If k > n, pad with n. Assign chunks 1:1 to tokens up to min(k, m).
        while len(src_bounds) < m:
            src_bounds.append(n)
        return src_bounds[:m]
    if n < k:
        # Not enough distinct source positions — pad boundaries to n.
        src_bounds = list(range(1, n + 1)) + [n] * (k - n)
    else:
        src_bounds = sorted(rng.sample(range(1, n + 1), k))

    # Split m target tokens into k groups: choose k-1 boundaries in 1..m-1.
    if k == 1:
        splits = []
    elif m - 1 < k - 1:
        splits = list(range(1, m))
    else:
        splits = sorted(rng.sample(range(1, m), k - 1))
    splits = [0] + splits + [m]

    commit = [n] * m
    for gi in range(k):
        for j in range(splits[gi], splits[gi + 1]):
            commit[j] = src_bounds[gi]
    return _enforce_monotone(commit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices",
                    default=str(REPO_ROOT / "results" / "phase1_tau_sweep" / "matrices.jsonl"))
    ap.add_argument("--taus", default="0.02,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_random_repeats", type=int, default=100,
                    help="Sample the random trace this many times per sentence "
                         "and average the resulting Pearson.")
    ap.add_argument("--output",
                    default=str(REPO_ROOT / "results" / "phase1_tau_sweep" / "random_floor.json"))
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    rng = random.Random(args.seed)

    per_sentence = []
    with open(args.matrices) as f:
        for line in f:
            per_sentence.append(json.loads(line))
    print(f"Loaded {len(per_sentence)} sentences from {args.matrices}")

    out = []
    for tau in taus:
        js_pearsons = []
        rand_pearsons_mean = []
        for rec in per_sentence:
            n = rec["n_src_tok"]
            m = rec["n_tgt_tok"]
            js_commit = commit_from_matrix(rec["matrix"], tau, n)
            k = chunk_count(js_commit)
            xs = [c / n for c in js_commit]
            ys = [(j + 1) / m for j in range(m)]
            js_p = pearson(xs, ys)
            js_pearsons.append(js_p)

            # Random floor at matched chunk count k.
            rp = []
            for _ in range(args.n_random_repeats):
                rc = random_commit(n, m, k, rng)
                xs_r = [c / n for c in rc]
                p = pearson(xs_r, ys)
                if not math.isnan(p):
                    rp.append(p)
            rand_pearsons_mean.append(sum(rp) / len(rp) if rp else float("nan"))

        def summarise(vals):
            finite = sorted(v for v in vals if not math.isnan(v))
            if not finite:
                return {"n_defined": 0, "n_nan": len(vals),
                        "min": None, "median": None, "max": None}
            return {"n_defined": len(finite), "n_nan": len(vals) - len(finite),
                    "min": finite[0], "median": finite[len(finite) // 2],
                    "max": finite[-1]}

        js_s = summarise(js_pearsons)
        rd_s = summarise(rand_pearsons_mean)
        row = {
            "tau": tau,
            "js": js_s,
            "random_at_matched_latency": rd_s,
            "js_beats_random_median": (
                (js_s["median"] is not None and rd_s["median"] is not None)
                and (js_s["median"] < rd_s["median"])
            ),
        }
        out.append(row)

    Path(args.output).write_text(json.dumps({"config": vars(args), "rows": out}, indent=2))
    print(f"Wrote {args.output}\n")

    print(f"{'tau':>6}  {'JS_med':>7}  {'JS_min':>7}  {'RD_med':>7}  {'RD_min':>7}  {'JS_beats_RD?':>13}")
    for row in out:
        js = row["js"]; rd = row["random_at_matched_latency"]
        def f(x): return f"{x:.3f}" if x is not None else "  --"
        print(f"{row['tau']:>6.2f}  {f(js['median']):>7}  {f(js['min']):>7}  "
              f"{f(rd['median']):>7}  {f(rd['min']):>7}  "
              f"{('YES' if row['js_beats_random_median'] else 'no'):>13}")


if __name__ == "__main__":
    main()
