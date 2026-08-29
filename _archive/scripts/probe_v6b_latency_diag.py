"""Diagnose whether v6b's latency conditioning actually works, or whether
the model commits early regardless and racks up BLEU via language-model priors.

Runs 3 checks per (direction, latency) pair:
  (1) g_words trajectory: does g grow monotonically as target emits, or
      does it plateau near 1 the whole target?
  (2) chunks_committed histogram at low vs high — does the model actually
      commit more frequently at low, or does it always produce 1-2 chunks?
  (3) hyp vs ref inspection: do the "low" hyps look like plausible
      translations of a source that has BARELY been read?
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EXTR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic")
LATS = ["low", "low_medium", "medium", "medium_high", "high"]
DIRS = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]


def load(lat: str, direction: str):
    fp = EXTR / f"flores_stream_v6b_checkargmax_{lat}_{direction}_n50.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def check1_g_trajectories(direction="en-vi"):
    print("\n" + "="*90)
    print(f"CHECK 1: g_words trajectories — direction={direction}")
    print("="*90)
    print("For each latency, print g_words for 3 sample sentences.")
    print("g_words[i] = # source words read when target word i was emitted.")
    print("If g_words stays near 1 throughout target, model is committing before reading source.\n")
    for lat in LATS:
        d = load(lat, direction)
        if d is None: continue
        per_sent = d.get("stream_stats", {}).get("per_sent", [])
        if not per_sent: continue
        print(f"--- {lat} ---")
        for i in [0, 10, 25]:
            if i >= len(per_sent): continue
            p = per_sent[i]
            g = p.get("g_words", [])
            src_w = p.get("src_words", 0)
            print(f"  sent{i}: src_words={src_w:>3}  y_len_g={len(g):>3}  al={p.get('al'):.2f}")
            gs = ",".join(str(x) for x in g[:20])
            more = f"...+{len(g)-20} more" if len(g)>20 else ""
            print(f"    g_words: [{gs}]{more}")


def check2_chunks_histogram(direction="en-vi"):
    print("\n" + "="*90)
    print(f"CHECK 2: chunks/sentence distribution — direction={direction}")
    print("="*90)
    print("If low ≈ high in chunk count, model isn't varying commit frequency with latency.\n")
    print(f"{'lat':<12} n_sents  chunks: mean  median   min  max  |  src_words_median  y_len_g_median")
    for lat in LATS:
        d = load(lat, direction)
        if d is None: continue
        per_sent = d.get("stream_stats", {}).get("per_sent", [])
        if not per_sent: continue
        ccs = [p.get("chunks", 0) for p in per_sent]
        srcs = sorted([p.get("src_words", 0) for p in per_sent])
        ylens = sorted([p.get("y_len_g", 0) for p in per_sent])
        import statistics as s
        median_src = srcs[len(srcs)//2]
        median_y = ylens[len(ylens)//2]
        print(f"  {lat:<10} {len(per_sent):>5}    {s.mean(ccs):>5.2f}  {s.median(ccs):>5.1f}  {min(ccs):>3}  {max(ccs):>3}  |  "
              f"{median_src:>15}  {median_y:>13}")


def check3_hyp_ref_inspection(direction="en-vi"):
    print("\n" + "="*90)
    print(f"CHECK 3: hyp vs ref at LOW latency — direction={direction}")
    print("="*90)
    print("If AL~1 means model read ~1 src word before emitting, is the hyp coherent?\n")
    d = load("low", direction)
    if d is None:
        print("  no low file for", direction); return
    hyps = d.get("hyps", [])
    refs = d.get("refs", [])
    per_sent = d.get("stream_stats", {}).get("per_sent", [])
    for i in [0, 5, 12, 25, 40]:
        if i >= len(hyps): continue
        al = per_sent[i].get("al") if i < len(per_sent) else None
        gw = per_sent[i].get("g_words", []) if i < len(per_sent) else []
        srcw = per_sent[i].get("src_words", 0) if i < len(per_sent) else 0
        chunks = per_sent[i].get("chunks", 0) if i < len(per_sent) else 0
        print(f"--- sent{i}: src_words={srcw}  chunks={chunks}  al={al}  g_first5={gw[:5]}")
        print(f"    HYP: {hyps[i][:200]!r}")
        print(f"    REF: {refs[i][:200]!r}")


def summary_all_directions():
    print("\n" + "="*90)
    print("SUMMARY: chunks-per-sent mean at each (lat, dir)")
    print("If the number changes with latency, model is adapting. If constant, it isn't.")
    print("="*90)
    print(f"{'dir':<8}  " + " ".join(f"{l:>12}" for l in LATS))
    for d in DIRS:
        row = [f"{d:<8}"]
        for lat in LATS:
            data = load(lat, d)
            if data is None:
                row.append(f"{'--':>12}"); continue
            ccs = [p.get("chunks", 0) for p in data.get("stream_stats", {}).get("per_sent", [])]
            src_words = [p.get("src_words", 0) for p in data.get("stream_stats", {}).get("per_sent", [])]
            if not ccs:
                row.append(f"{'--':>12}"); continue
            import statistics as s
            mean_c = s.mean(ccs)
            mean_src = s.mean(src_words) if src_words else 0
            # Report ratio too: chunks per source word
            cell = f"{mean_c:5.1f}c/{mean_src:.0f}w"
            row.append(f"{cell:>12}")
        print(" ".join(row))


if __name__ == "__main__":
    # Focus on en-vi (highest BLEU + lowest AL — most suspicious)
    check1_g_trajectories("en-vi")
    check2_chunks_histogram("en-vi")
    check3_hyp_ref_inspection("en-vi")
    # Also check en-ru (AL 0.88 at low — most extreme)
    print("\n\n" + "#"*90)
    print("# Also probing en-ru — most extreme AL (0.88 at low)")
    print("#"*90)
    check1_g_trajectories("en-ru")
    check2_chunks_histogram("en-ru")
    check3_hyp_ref_inspection("en-ru")
    # Summary across all directions
    summary_all_directions()
