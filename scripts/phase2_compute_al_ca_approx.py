"""
Compute a corpus-level AL-CA approximation from existing full_stream_*.json
outputs. Proper per-token AL-CA needs torch.cuda.Event instrumentation
(Layer 3, not yet run); this gives a first-order estimate assuming
uniform per-token wall time within a run.

AL-CA(g) = (1/tau) * sum [g(i) + T(i)*K - (i-1)*|X|/|Y|]
       where T(i) = wall time to emit target token i (seconds),
             K = assumed source-word arrival rate (default 1/s → wall time
                 in seconds ≈ source-word-equivalents).

Approximation: T(i) ≈ total_wall / total_target_words (uniform across
tokens within a run, uniform across sentences within a run). This
underestimates AL-CA on sentences with above-average generation cost
and overestimates on faster ones — but the corpus mean should be close
to the true corpus AL-CA.

Reads results/phase2/extrinsic/full_stream_*.json and prints a table
mirroring EAST Table 3's shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_and_compute(path: Path, source_rate_words_per_sec: float = 1.0):
    d = json.loads(path.read_text())
    n_sents = d["n_sentences"]
    wall = d["wall_time_sec"]
    per_sent = d["stream_stats"].get("per_sent", [])
    if not per_sent:
        return None
    total_tgt_words = sum(p["y_len_g"] for p in per_sent)
    if total_tgt_words == 0:
        return None
    sec_per_tgt_word = wall / total_tgt_words
    source_word_lag_per_tgt = sec_per_tgt_word * source_rate_words_per_sec

    al_mean = d["al_mean"]
    # AL-CA adds the wall-time lag as source-word-equivalents. Uniform-per-
    # token approximation ⇒ AL-CA = AL_mean + (sum_{i=1..tau} i * lag/tau) / 1
    # ≈ AL_mean + lag * (tau+1)/2. Approximate tau ~ y_len_g (typical) and
    # take corpus average.
    # Simpler bound: AL-CA ≈ AL_mean + lag * avg_tgt_words / 2 (average tau).
    avg_tgt_words = total_tgt_words / len(per_sent)
    al_ca_approx = al_mean + source_word_lag_per_tgt * (avg_tgt_words / 2)

    return {
        "bleu": d["bleu"],
        "al_mean": al_mean,
        "al_ca_approx": al_ca_approx,
        "sec_per_tgt_word": sec_per_tgt_word,
        "wall_total": wall,
        "avg_tgt_words": avg_tgt_words,
        "chunks_per_sent": d["stream_stats"]["chunks_per_sent_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrinsic_dir", type=Path,
                    default=Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic"))
    args = ap.parse_args()

    # Fixed set of runs we care about. Only live arm is OT-SFT (paths are
    # results/phase2/extrinsic/full_stream_<policy>_n10k.json under the new
    # naming after cond-A/cond-C removal 2026-08-18 late).
    runs = []
    for policy in ["waitk1", "waitk3", "waitk5", "waitk7", "waitk9", "waitk11", "checkargmax"]:
        p = args.extrinsic_dir / f"full_stream_{policy}_n10k.json"
        if p.exists():
            runs.append((policy, p))

    print("=== AL-CA approximation (uniform-per-token, source_rate=1 word/s) ===\n")
    print(f"{'Policy':<14} {'BLEU':<8} {'AL':<8} {'AL-CA':<8} {'sec/tgt':<10} {'chunks/sent':<12}")
    print("-" * 66)
    for policy, p in runs:
        r = load_and_compute(p)
        if r is None:
            print(f"{policy:<14} (missing per_sent trace)")
            continue
        print(f"{policy:<14} {r['bleu']:<8.2f} {r['al_mean']:<8.2f} {r['al_ca_approx']:<8.2f} {r['sec_per_tgt_word']:<10.4f} {r['chunks_per_sent']:<12.2f}")

    print()
    print("Caveats:")
    print("  - AL-CA_approx assumes uniform per-token wall time; real distribution has a tail.")
    print("  - source_rate = 1 word/sec is a convention (EAST §5.2); adjust for real deployment.")
    print("  - Proper AL-CA requires torch.cuda.Event per generation step — rerun with Layer 3.")


if __name__ == "__main__":
    main()
