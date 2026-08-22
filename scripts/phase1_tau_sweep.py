"""
Tau sweep on the Phase-1 smoke set.

Annotates once per sentence with return_full_matrix=True (records every
D(P_full[j], P_pre[i][j]) — one GPU forward per prefix length), then
derives commit points at multiple tau values offline. Reports for each
tau:
  * fire_fraction — sentences where >=1 target token committed before n
  * committed_fraction — mean (over sentences) of fraction of target
    tokens that committed before n
  * chunks_per_sentence_mean
  * pearson_i_over_n_vs_j_over_m median / min / max / #NaN
  * mean fired divergence (informational — where tau bites for JS)

Output:
  results/phase1_tau_sweep/matrices.jsonl — per-sentence full matrix
  results/phase1_tau_sweep/summary.json   — sweep table

The matrices file lets us swap the criterion (KL / OT later) or run
finer tau grids without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.annotator.annotate import (
    _chunks_from_commit,
    _enforce_monotone,
    annotate_pair,
)
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT


CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


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


def pick_sentences(rows, n_total: int, seed: int):
    import random
    rng = random.Random(seed)
    by_latency = {}
    for r in rows:
        by_latency.setdefault(r["latency"], []).append(r)
    per = n_total // 3
    picked = []
    for lat in ["low", "medium", "high"]:
        pool = by_latency.get(lat, [])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    remainder = n_total - len(picked)
    if remainder > 0:
        picked.extend(by_latency["medium"][per : per + remainder])
    return picked


def commit_from_matrix(matrix, tau, n):
    """Given the (n, m) matrix (rows = source prefix length i=1..n, cols = target token j),
    derive commit points per target token: first i for which div < tau. Fallback n."""
    if not matrix:
        return []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sentences", type=int, default=50)
    ap.add_argument("--criterion", type=str, default="js", choices=["js", "kl", "ot"])
    ap.add_argument("--taus", type=str, default="0.02,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase1_tau_sweep")
    ap.add_argument("--model_path", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--prompt_mode", type=str, default="raw", choices=["raw", "chat"],
                    help="raw: source_prefix + newline; "
                         "chat: Gemma-style chat template with translation instruction "
                         "(required for instruction-tuned backbones like gemma-4-*-it).")
    ap.add_argument("--record_entropy", action="store_true",
                    help="Also record H(P_pre[i][j]) matrix and H(P_full[j]) — "
                         "enables the entropy-only ablation criterion offline.")
    ap.add_argument("--indices_file", type=Path, default=None,
                    help="Path to a JSON file with an 'indices' list — use these "
                         "corpus indices exactly (overrides --n_sentences balanced-"
                         "latency sampling). Used by Gate-1 stratified-by-reordering "
                         "runs (see phase1_precompute_gpt4_pearson.py).")
    ap.add_argument("--input_json", type=Path, default=None,
                    help="Path to a pre-built row JSON (list of {source,target,"
                         "src_lang,tgt_lang,index[,latency]}). Overrides CORPUS "
                         "and skips latency-balanced sampling — use every row in "
                         "input_json. Enables multilingual pipeline: sample once "
                         "per direction, then annotate each direction file.")
    ap.add_argument("--resume", action="store_true",
                    help="Read output_dir/matrices.jsonl if it exists, skip already-"
                         "processed indices, append to the same file. Enables sharded "
                         "annotation across multiple PBS submissions. Line-flushes per "
                         "sentence so a mid-sentence kill loses at most one row.")
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    print(f"Tau grid: {taus}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path)
    if getattr(cfg, "model_type", None) == "gemma3n":
        # Gemma-4-E4B (gemma3n): skip vision tower, load text-only.
        from transformers import Gemma3nForCausalLM
        print("  (model_type=gemma3n; loading text-only Gemma3nForCausalLM)")
        model = Gemma3nForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    if args.input_json is not None:
        with open(args.input_json) as f:
            rows = json.load(f)
        print(f"Input pool: {args.input_json} ({len(rows):,} rows)")
        picks = rows  # use every row — the pool is pre-sampled/pre-filtered
    else:
        with open(CORPUS) as f:
            rows = json.load(f)
        print(f"Corpus: {len(rows):,} rows")
        if args.indices_file is not None:
            idx_spec = json.loads(args.indices_file.read_text())
            wanted = set(idx_spec["indices"])
            by_idx = {r["index"]: r for r in rows}
            picks = [by_idx[i] for i in sorted(wanted) if i in by_idx]
            print(f"Using {len(picks)} indices from {args.indices_file} "
                  f"(bin thresholds: {idx_spec.get('thresholds', {})})")
        else:
            picks = pick_sentences(rows, args.n_sentences, args.seed)
    kept = []
    for r in picks:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src <= args.max_src_tokens:
            kept.append(r)
    print(f"Kept {len(kept)}/{len(picks)} sentences after max_src_tokens={args.max_src_tokens}")

    matrices_path = args.output_dir / "matrices.jsonl"
    done_marker = args.output_dir / "DONE"
    resume_marker = args.output_dir / "NEEDS_RESUME"

    # Resume mode: read existing matrices.jsonl, get processed indices, skip those.
    processed = set()
    if args.resume and matrices_path.exists():
        with open(matrices_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed.add(rec["index"])
                except Exception:
                    pass
        print(f"Resume mode: {len(processed)} indices already in matrices.jsonl; skipping")
    remaining = [r for r in kept if r["index"] not in processed]
    print(f"To process this shard: {len(remaining)} sentences")

    if not remaining:
        # Fully done — write DONE and load per_sentence for summary.
        print("All indices already processed — writing DONE marker and computing summary.")
        done_marker.write_text("done\n")
        if resume_marker.exists():
            resume_marker.unlink()
        # Reload for summary.
        per_sentence = []
        with open(matrices_path) as f:
            for line in f:
                per_sentence.append(json.loads(line))
    else:
        per_sentence = []
        # Load previous rows (for summary later) — cheap given they're already on disk.
        if args.resume and matrices_path.exists():
            with open(matrices_path) as f:
                for line in f:
                    per_sentence.append(json.loads(line))
        t_annot = time.time()
        mode = "a" if args.resume else "w"
        with open(matrices_path, mode, buffering=1) as f:
            for k, r in enumerate(remaining):
                t0 = time.time()
                try:
                    # tau=0 forces the criterion never to fire in the search loop
                    # itself; we set return_full_matrix and derive commits offline.
                    ann = annotate_pair(
                        model=model, tokenizer=tokenizer,
                        source=r["source"], target=r["target"],
                        src_lang=r["src_lang"], tgt_lang=r["tgt_lang"],
                        latency=r["latency"],
                        tau=0.0, criterion_name=args.criterion,
                        return_full_matrix=True,
                        record_entropy=args.record_entropy,
                        prompt_mode=args.prompt_mode,
                    )
                except Exception as e:
                    print(f"  [{k+1}/{len(remaining)}] idx={r['index']} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                dt = time.time() - t0
                rec = {
                    "index": r["index"],
                    "latency": r["latency"],
                    "n_src_tok": ann.n_src_tok,
                    "n_tgt_tok": ann.n_tgt_tok,
                    "gpt4_source_chunks": r["source_chunks"],
                    "gpt4_target_chunks": r["target_chunks"],
                    "matrix": ann.divergence_matrix,
                    "dt_sec": dt,
                }
                if args.record_entropy:
                    rec["entropy_matrix"] = ann.entropy_matrix
                    rec["entropy_full"] = ann.entropy_full
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                import os as _os; _os.fsync(f.fileno())
                per_sentence.append(rec)
                if k < 5 or (k + 1) % 10 == 0:
                    print(f"  [{k+1}/{len(remaining)}] idx={r['index']} lat={r['latency']} "
                          f"n={ann.n_src_tok} m={ann.n_tgt_tok} dt={dt:.1f}s", flush=True)

        # After the loop: check if we've caught up to `kept`. If yes, DONE; if
        # not (killed by walltime), NEEDS_RESUME.
        processed_now = set()
        with open(matrices_path) as f:
            for line in f:
                try:
                    processed_now.add(json.loads(line)["index"])
                except Exception:
                    pass
        kept_indices = {r["index"] for r in kept}
        if kept_indices - processed_now:
            outstanding = len(kept_indices - processed_now)
            resume_marker.write_text(f"outstanding={outstanding}\n")
            if done_marker.exists():
                done_marker.unlink()
            print(f"\n{outstanding} sentences remain — wrote NEEDS_RESUME marker.",
                  flush=True)
        else:
            done_marker.write_text("done\n")
            if resume_marker.exists():
                resume_marker.unlink()
            print(f"\nAll {len(kept_indices)} sentences processed — wrote DONE marker.",
                  flush=True)

    total_dt = time.time() - t_annot if remaining else 0
    print(f"\nAnnotated {len(per_sentence)}/{len(kept)} in {total_dt:.1f}s "
          f"({total_dt/max(len(per_sentence),1):.1f}s/sentence)")

    # Offline sweep over tau on the recorded matrices.
    sweep = []
    for tau in taus:
        fire_bool = []      # per sentence: did any target token commit early?
        committed_frac = [] # per sentence: fraction of tgt tokens committed early
        chunks_ours = []    # per sentence: len(source_chunks_ours)
        chunks_gpt4 = []    # per sentence: len(source_chunks_gpt4)
        pearsons = []       # per sentence: Pearson(i*/n, j/m) if defined
        min_fired = []      # per sentence: mean fired divergence for tokens that fired
        for rec in per_sentence:
            n = rec["n_src_tok"]
            m = rec["n_tgt_tok"]
            commit = commit_from_matrix(rec["matrix"], tau, n)
            fired = [c for c in commit if c < n]
            fire_bool.append(len(fired) > 0)
            committed_frac.append(len(fired) / m if m > 0 else 0.0)
            # chunk counts
            xs = [c / n for c in commit]
            ys = [(j + 1) / m for j in range(m)]
            pearsons.append(pearson(xs, ys))
            chunks_gpt4.append(len(rec["gpt4_source_chunks"]))
            # count run-length groups over commit for our chunk count
            groups = 0
            j = 0
            while j < m:
                k = j
                while k + 1 < m and commit[k + 1] == commit[j]:
                    k += 1
                groups += 1
                j = k + 1
            chunks_ours.append(groups)
            # min fired divergence per sentence (informational)
            fired_ds = []
            for j in range(m):
                if commit[j] < n:
                    d = rec["matrix"][commit[j] - 1][j]
                    if not math.isnan(d):
                        fired_ds.append(d)
            min_fired.append(sum(fired_ds) / len(fired_ds) if fired_ds else float("nan"))

        finite_p = [p for p in pearsons if not math.isnan(p)]
        finite_p_sorted = sorted(finite_p)
        n_finite = len(finite_p)
        sweep_row = {
            "tau": tau,
            "fire_fraction": sum(fire_bool) / max(len(fire_bool), 1),
            "mean_committed_fraction": sum(committed_frac) / max(len(committed_frac), 1),
            "chunks_per_sentence_mean_ours": sum(chunks_ours) / max(len(chunks_ours), 1),
            "chunks_per_sentence_mean_gpt4": sum(chunks_gpt4) / max(len(chunks_gpt4), 1),
            "pearson_i_over_n_vs_j_over_m": {
                "n_defined": n_finite,
                "n_nan": len(pearsons) - n_finite,
                "min": finite_p_sorted[0] if finite_p_sorted else None,
                "median": finite_p_sorted[n_finite // 2] if finite_p_sorted else None,
                "max": finite_p_sorted[-1] if finite_p_sorted else None,
            },
        }
        sweep.append(sweep_row)

    summary = {
        "config": {
            "n_sentences_requested": args.n_sentences,
            "n_sentences_kept": len(kept),
            "n_sentences_annotated": len(per_sentence),
            "criterion": args.criterion,
            "seed": args.seed,
            "model_path": args.model_path,
            "max_src_tokens": args.max_src_tokens,
            "taus": taus,
            "prompt_mode": args.prompt_mode,
            "record_entropy": args.record_entropy,
            "indices_file": str(args.indices_file) if args.indices_file else None,
        },
        "sweep": sweep,
        "env": {
            "device": device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": os.popen("git -C " + str(REPO_ROOT) + " rev-parse HEAD").read().strip(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Tau sweep ===")
    print(f"{'tau':>6}  {'fire%':>6}  {'commit%':>8}  {'ours_chunks':>11}  {'gpt4_chunks':>11}  "
          f"{'pearson_med':>11}  {'pearson_min':>11}  {'#nan_pear':>9}")
    for row in sweep:
        p = row["pearson_i_over_n_vs_j_over_m"]
        pm = f"{p['median']:.3f}" if p["median"] is not None else "  --"
        pmin = f"{p['min']:.3f}" if p["min"] is not None else "  --"
        print(f"{row['tau']:>6.2f}  {row['fire_fraction']*100:>5.1f}%  "
              f"{row['mean_committed_fraction']*100:>7.1f}%  "
              f"{row['chunks_per_sentence_mean_ours']:>11.2f}  "
              f"{row['chunks_per_sentence_mean_gpt4']:>11.2f}  "
              f"{pm:>11}  {pmin:>11}  {p['n_nan']:>9d}")

    print("\nPHASE 1 TAU SWEEP COMPLETE")


if __name__ == "__main__":
    main()
