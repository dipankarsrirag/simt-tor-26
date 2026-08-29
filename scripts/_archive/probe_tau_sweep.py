"""Sweep tau over cached OT matrices to characterize chunk-count operating curve.

For each tau in a grid, replays commit_from_matrix on all annotated rows for
one direction and prints:
  - fraction of rows that collapse to 1 chunk (unusable)
  - fraction that stay < 3 chunks (would be labeled "high")
  - mean / median chunks-per-sent (excluding 1-chunk collapses)
  - chunks_per_source_word ratio (comparable across sentence lengths)

Target: find tau where mean chunks/sent ≈ 5-8 (EAST's low-latency regime)
without collapsing more than ~5% of rows.

JS divergence bound: [0, log(2)] ≈ [0, 0.693]. Values used:
  0.03 (~5%   of max) — very strict
  0.05 (~7%   of max)
  0.10 (~14%  of max)
  0.15 (~22%  of max)
  0.20 (~29%  of max)
  0.30 (~43%  of max) — current primary
  0.50 (~72%  of max)
"""
from __future__ import annotations

import json
import math
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.annotate import _enforce_monotone

MATRICES = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/annot_ot_multi_de-en/matrices.jsonl")
TAUS = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


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


def chunks(commit):
    if not commit: return 0
    g, j, m = 0, 0, len(commit)
    while j < m:
        k = j
        while k + 1 < m and commit[k + 1] == commit[j]:
            k += 1
        g += 1
        j = k + 1
    return g


def main():
    print(f"Loading matrices from {MATRICES.name} ...")
    rows = []
    with open(MATRICES) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"  {len(rows)} rows")
    print()

    print(f"{'tau':>6}  {'%coll1':>7}  {'%<=3':>7}  {'mean_c':>7}  {'med_c':>6}  "
          f"{'c/srcw':>7}  {'p10_c':>6}  {'p90_c':>6}")
    print("─" * 70)

    for tau in TAUS:
        cs, cs_ok = [], []  # all chunks, chunks with > 1
        cpw = []  # chunks per source word
        n1 = 0; nle3 = 0
        for r in rows:
            n = r["n_src_tok"]
            commit = commit_from_matrix(r["matrix"], tau, n)
            c = chunks(commit)
            cs.append(c)
            if c == 1: n1 += 1
            else: cs_ok.append(c)
            if c <= 3: nle3 += 1
            if n > 0:
                cpw.append(c / n)
        if not cs_ok:
            print(f"{tau:>6.3f}  {'100%':>7}  ---")
            continue
        mean_c = stats.mean(cs_ok)
        med_c = stats.median(cs_ok)
        cs_sorted = sorted(cs_ok)
        p10 = cs_sorted[int(len(cs_sorted)*0.10)]
        p90 = cs_sorted[int(len(cs_sorted)*0.90)]
        print(f"{tau:>6.3f}  {n1*100/len(rows):>6.1f}%  {nle3*100/len(rows):>6.1f}%  "
              f"{mean_c:>7.2f}  {med_c:>6.1f}  {stats.mean(cpw):>7.3f}  {p10:>6d}  {p90:>6d}")

    print()
    print("Legend: %coll1=% of rows collapsed to 1 chunk (unusable — drop);  "
          "%<=3=% would be 'high' latency in EAST scheme;")
    print("        mean_c/med_c=chunks per sentence among rows with >1 chunk;  "
          "c/srcw=chunks per source word (density);")
    print("        p10/p90=10th/90th %ile chunk counts (spread).")


if __name__ == "__main__":
    main()
