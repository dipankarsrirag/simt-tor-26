"""
Pilot for lookahead_k choice.

Runs annotate_pair on the first N (default 100) de-en sentences from the
multilingual source pool at k ∈ {0, 1, 2, 3}, using the SAME sentences
across k values so per-k comparisons are strictly paired.

Reports per k:
  * mean chunks per sentence at each tau in the v2bal grid {0.20, 0.40, 0.60}
  * fire fraction at tau=0.30 (does the criterion actually decide?)
  * commit-point shift vs k=0 baseline (mean |Δcommit_j| across target tokens)
  * fraction of sentences whose k=0 vs k>0 chunk count differs by >=1

Feeds the pick-k decision before the 80K full run. Writes both a human
summary and a per-sentence JSONL for later inspection.

Usage on PBS:
  python -u scripts/probe_lookahead_pilot.py --n 100 --k_grid 0,1,2,3

~7 min on 1×H200 for n=100 (~1s/sentence × 4 k values, roughly 1× cost each).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from src.annotator.annotate import annotate_pair, _enforce_monotone
from src.constants import PRIMARY_BACKBONE, REPO_ROOT


def commit_from_matrix(matrix, tau, n):
    """First i with div < tau, else n (1-indexed prefix length)."""
    if not matrix or not matrix[0]:
        return []
    m = len(matrix[0])
    commit = [n] * m
    for j in range(m):
        for i in range(len(matrix)):
            d = matrix[i][j]
            if not math.isnan(d) and d < tau:
                commit[j] = i + 1
                break
    return _enforce_monotone(commit)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k_grid", type=str, default="0,1,2,3")
    ap.add_argument("--tau_grid", type=str, default="0.20,0.30,0.40,0.60")
    ap.add_argument("--direction", type=str, default="de-en")
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "probe_lookahead_pilot")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.k_grid.split(",")]
    taus = [float(x) for x in args.tau_grid.split(",")]

    pool_path = REPO_ROOT / "results" / "phase2" / \
        "multilingual_source_pool_v5_per_direction" / f"{args.direction}.json"
    with open(pool_path) as f:
        pool = json.load(f)
    print(f"Loaded {len(pool)} rows from {pool_path}")

    # Load model once
    device = "cuda"
    print(f"Loading {PRIMARY_BACKBONE} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(PRIMARY_BACKBONE))
    cfg = AutoConfig.from_pretrained(str(PRIMARY_BACKBONE))
    if getattr(cfg, "model_type", None) == "gemma3n":
        from transformers import Gemma3nForCausalLM
        model = Gemma3nForCausalLM.from_pretrained(
            str(PRIMARY_BACKBONE), dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(PRIMARY_BACKBONE), dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
    model.eval()

    # Filter pool to max_src_tokens
    kept = []
    for r in pool:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src <= args.max_src_tokens:
            kept.append(r)
        if len(kept) >= args.n:
            break
    print(f"Kept {len(kept)}/{args.n} sentences after max_src_tokens={args.max_src_tokens}")

    # For each sentence: annotate at each k, keep matrices in memory
    per_sent = []  # per-sentence records
    for si, r in enumerate(kept):
        rec = {"idx": r.get("index", si), "src_len_words": len(r["source"].split()),
               "runs": {}}
        for k in ks:
            t0 = time.time()
            try:
                ann = annotate_pair(
                    model=model, tokenizer=tokenizer,
                    source=r["source"], target=r["target"],
                    src_lang=r["src_lang"], tgt_lang=r["tgt_lang"],
                    latency=r.get("latency", "medium"),
                    tau=0.0, criterion_name="ot",
                    return_full_matrix=True,
                    prompt_mode="raw",
                    lookahead_k=k,
                )
            except Exception as e:
                print(f"  sent {si} k={k} FAILED: {type(e).__name__}: {e}", flush=True)
                rec["runs"][str(k)] = {"error": f"{type(e).__name__}: {e}"}
                continue
            dt = time.time() - t0
            n = ann.n_src_tok
            m = ann.n_tgt_tok
            per_tau = {}
            for tau in taus:
                commit = commit_from_matrix(ann.divergence_matrix, tau, n)
                fire = sum(1 for c in commit if c < n)
                chunks = len(set(commit))
                per_tau[f"{tau:.2f}"] = {
                    "commit": commit,
                    "fire_frac": fire / max(len(commit), 1),
                    "chunks": chunks,
                }
            rec["runs"][str(k)] = {
                "n": n, "m": m, "dt": dt,
                "per_tau": per_tau,
            }
        per_sent.append(rec)
        if (si + 1) % 10 == 0 or si < 3:
            print(f"  [{si+1}/{len(kept)}] "
                  f"chunks@tau=0.30 across k: "
                  f"{[rec['runs'].get(str(k), {}).get('per_tau', {}).get('0.30', {}).get('chunks', '?') for k in ks]}",
                  flush=True)

    # Aggregate across sentences
    print("\n=== Aggregate (mean over sentences) ===")
    header = f"{'k':>3}  " + "  ".join(f"tau={tau:.2f}: {'chunks':>7} {'fire%':>6}" for tau in taus)
    print(header)
    aggregate = {}
    for k in ks:
        row = {"k": k, "per_tau": {}}
        for tau in taus:
            chunk_vals = []
            fire_vals = []
            for rec in per_sent:
                run = rec["runs"].get(str(k), {})
                pt = run.get("per_tau", {}).get(f"{tau:.2f}")
                if pt:
                    chunk_vals.append(pt["chunks"])
                    fire_vals.append(pt["fire_frac"])
            mean_chunks = sum(chunk_vals) / max(len(chunk_vals), 1)
            mean_fire = sum(fire_vals) / max(len(fire_vals), 1)
            row["per_tau"][f"{tau:.2f}"] = {
                "mean_chunks": mean_chunks,
                "mean_fire_frac": mean_fire,
                "n_samples": len(chunk_vals),
            }
        aggregate[str(k)] = row
        print(f"{k:>3}  " + "  ".join(
            f"           {row['per_tau'][f'{tau:.2f}']['mean_chunks']:>7.2f} "
            f"{row['per_tau'][f'{tau:.2f}']['mean_fire_frac']*100:>5.1f}%"
            for tau in taus
        ))

    # Commit-shift vs k=0 (paired within sentence at tau=0.30)
    print("\n=== Commit-shift vs k=0 (tau=0.30, per-sentence paired) ===")
    print(f"{'k':>3}  {'mean|Δi|':>10}  {'frac shifted':>13}  {'chunk count Δmean':>18}")
    shift_summary = {}
    for k in ks:
        if k == 0:
            continue
        deltas = []
        shifted = 0
        chunk_deltas = []
        total = 0
        for rec in per_sent:
            r0 = rec["runs"].get("0", {}).get("per_tau", {}).get("0.30")
            rk = rec["runs"].get(str(k), {}).get("per_tau", {}).get("0.30")
            if not r0 or not rk:
                continue
            c0 = r0["commit"]
            ck = rk["commit"]
            if len(c0) != len(ck):
                continue
            total += 1
            per_j = [abs(a - b) for a, b in zip(c0, ck)]
            if per_j:
                deltas.append(sum(per_j) / len(per_j))
            if any(d > 0 for d in per_j):
                shifted += 1
            chunk_deltas.append(rk["chunks"] - r0["chunks"])
        mean_delta = sum(deltas) / max(len(deltas), 1)
        frac_shifted = shifted / max(total, 1)
        chunk_dmean = sum(chunk_deltas) / max(len(chunk_deltas), 1)
        shift_summary[str(k)] = {
            "mean_abs_delta_commit": mean_delta,
            "frac_sentences_shifted": frac_shifted,
            "chunk_count_delta_mean": chunk_dmean,
            "n": total,
        }
        print(f"{k:>3}  {mean_delta:>10.3f}  {frac_shifted*100:>12.1f}%  {chunk_dmean:>+18.3f}")

    out = {
        "config": vars(args) | {"model_path": str(PRIMARY_BACKBONE)},
        "aggregate": aggregate,
        "shift_vs_k0_tau0p30": shift_summary,
    }
    # dataclass paths as str
    out["config"]["out_dir"] = str(out["config"]["out_dir"])
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=2))

    per_sent_path = args.out_dir / "per_sentence.jsonl"
    with open(per_sent_path, "w") as f:
        for rec in per_sent:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWritten: {args.out_dir / 'summary.json'} + per_sentence.jsonl")


if __name__ == "__main__":
    main()
