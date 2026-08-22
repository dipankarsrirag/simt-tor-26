"""Compute DAL (Cherry & Foster 2019) from existing per-sent stream traces.

DAL (Differentiable Average Lagging) fixes AL's multi-word-WRITE-chunk
artifact by enforcing a minimum monotonic increment per target position:

    g'(0)  = 0
    g'(i)  = max(g(i), g'(i-1) + |X|/|Y|)      for i = 1..|Y|
    DAL(g) = (1/|Y|) * Σ_{i=1..|Y|} [g'(i) - (i-1) * |X|/|Y|]

Interpretation: if a chunk emits N target words at the same g value, AL
treats them as instantaneous (lag drops for later positions in the burst);
DAL "spreads" them across a minimum-slope schedule so each target word
contributes at least ratio source-words of lag.

Reads all `flores_stream_v6b_checkargmax_*_n50.json` and prints a
grid of BLEU / AL / LAAL / DAL per (dir, latency).
"""
from __future__ import annotations

import glob
import json
import os
import statistics as stats

EXTR = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic"
LAT = ["low", "low_medium", "medium", "medium_high", "high"]
DIR = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]


def compute_dal(g_words, x_len, y_len):
    if not g_words or x_len == 0 or y_len == 0:
        return None
    ratio = x_len / y_len
    g_prime = 0.0
    total_lag = 0.0
    for i, g in enumerate(g_words, start=1):
        g_prime = max(g, g_prime + ratio)
        total_lag += g_prime - (i - 1) * ratio
    return total_lag / len(g_words)


def main():
    data = {}
    for f in sorted(glob.glob(f"{EXTR}/flores_stream_v6b_checkargmax_*_n50.json")):
        d = json.loads(open(f).read())
        name = os.path.basename(f).replace(".json", "")
        parts = name.split("_")
        n_idx = parts.index("n50")
        dir_str = parts[n_idx - 1]
        lat_str = "_".join(parts[4:n_idx - 1])
        per_sent = d.get("stream_stats", {}).get("per_sent", [])
        dal_vals = [
            compute_dal(p["g_words"], p["src_words"], p["y_len_g"])
            for p in per_sent
            if p.get("g_words") and p.get("src_words") and p.get("y_len_g")
        ]
        dal_vals = [v for v in dal_vals if v is not None]
        dal_mean = stats.mean(dal_vals) if dal_vals else None
        data[(lat_str, dir_str)] = (
            d["bleu"], d.get("al_mean"), d.get("laal_mean"), dal_mean
        )

    W = 27
    print()
    print("┌" + "─" * 9 + "┬" + "┬".join("─" * W for _ in LAT) + "┐")
    print("│ " + "dir".ljust(7) + " │" + "│".join(f" {l:<{W-2}} " for l in LAT) + "│")
    print("│ " + " ".ljust(7) + " │" + "│".join(f" {'BLEU / AL / LAAL / DAL':<{W-2}} " for _ in LAT) + "│")
    print("├" + "─" * 9 + "┼" + "┼".join("─" * W for _ in LAT) + "┤")
    for d in DIR:
        parts = ["│ " + d.ljust(7) + " │"]
        for lat in LAT:
            if (lat, d) in data:
                b, al, laal, dal = data[(lat, d)]
                cell = f"{b:5.2f}/{al:4.2f}/{laal:4.2f}/{dal:5.2f}"
            else:
                cell = "  --"
            parts.append(f" {cell:<{W-2}} │")
        print("".join(parts))
    print("└" + "─" * 9 + "┴" + "┴".join("─" * W for _ in LAT) + "┘")
    print()
    print("AL   = Ma 2019 (truncated at source-exhaustion; underestimates lag on chunky/over-long output)")
    print("LAAL = Papi 2022 (sums over ALL target words; length-adaptive)")
    print("DAL  = Cherry+Foster 2019 (enforces min slope; corrects multi-word-WRITE-chunk artifact)")


if __name__ == "__main__":
    main()
