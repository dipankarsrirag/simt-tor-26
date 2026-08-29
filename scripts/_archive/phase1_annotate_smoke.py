"""
Phase 1 smoke: annotate 20 sentences from SiMT-De-En-660K with the
Gemma-4-E2B annotator (JS-divergence commit criterion) and run every
sanity check from METHOD §8.

Outputs:
  * results/phase1_smoke_js/annotations.jsonl — per-sentence trace + chunks
  * results/phase1_smoke_js/summary.json      — aggregates + gate signals
  * stdout report suitable for LOG.md pasting

Gate signals:
  * i*[j]/n vs j/m Pearson correlation — if uniformly > 0.99 across
    sentences, positional-degeneracy is present and the criterion has
    no signal (METHOD §8).
  * Chunk-count agreement with shipped GPT-4 chunks (sanity, not a
    metric): comparing means gives a first impression of whether ours
    is coarser/finer than the baseline.
  * Fraction of sentences where the criterion never fires (fallback to
    commit=n) — high = tau too tight or oracle too weak.
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

from src.annotator.annotate import annotate_pair
from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT


CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def pearson(xs: list[float], ys: list[float]) -> float:
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


def pick_sentences(rows: list[dict], n_total: int, seed: int) -> list[dict]:
    """Pick n_total sentences balanced across the three latency levels.
    Preserve indices for reproducibility."""
    import random
    rng = random.Random(seed)
    by_latency: dict[str, list[dict]] = {}
    for r in rows:
        by_latency.setdefault(r["latency"], []).append(r)
    picked = []
    per = n_total // 3
    for lat in ["low", "medium", "high"]:
        pool = by_latency.get(lat, [])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    remainder = n_total - len(picked)
    if remainder > 0:
        picked.extend(by_latency["medium"][per : per + remainder])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sentences", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--criterion", type=str, default="js", choices=["js", "kl"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase1_smoke_js")
    ap.add_argument("--model_path", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--max_src_tokens", type=int, default=80,
                    help="Skip sentences with more source tokens than this "
                         "(smoke only — annotator cost is O(n) forwards).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Loading corpus {CORPUS} ...")
    with open(CORPUS) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} rows")

    picks = pick_sentences(rows, args.n_sentences, args.seed)
    print(f"Picked {len(picks)} sentences (balanced across latency levels).")

    # Filter length after picking so we still balance latency.
    kept = []
    for r in picks:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src <= args.max_src_tokens:
            kept.append(r)
    print(f"After max_src_tokens={args.max_src_tokens} filter: {len(kept)} kept.")

    annotations_path = args.output_dir / "annotations.jsonl"
    summary_path = args.output_dir / "summary.json"
    per_sentence: list[dict] = []
    correlations: list[float] = []
    fire_counts = 0
    all_finite_div = 0
    chunk_agree_counts: list[tuple[int, int]] = []  # (ours, gpt4)

    t_annot = time.time()
    with open(annotations_path, "w") as f:
        for k, r in enumerate(kept):
            t0 = time.time()
            try:
                ann = annotate_pair(
                    model=model, tokenizer=tokenizer,
                    source=r["source"], target=r["target"],
                    src_lang=r["src_lang"], tgt_lang=r["tgt_lang"],
                    latency=r["latency"],
                    tau=args.tau,
                    criterion_name=args.criterion,
                )
            except Exception as e:
                print(f"  [{k+1}/{len(kept)}] idx={r['index']} FAILED: {type(e).__name__}: {e}")
                continue
            dt = time.time() - t0

            # Sanity: positional-degeneracy check (i*[j]/n vs j/m).
            n = ann.n_src_tok
            m = ann.n_tgt_tok
            x = [ann.commit_source_tok_idx[j] / n for j in range(m)]
            y = [(j + 1) / m for j in range(m)]
            r_pearson = pearson(x, y)
            correlations.append(r_pearson)

            # Did the criterion ever fire (i.e., not all fallback-to-n)?
            n_fired = sum(1 for j in range(m) if ann.commit_source_tok_idx[j] < n)
            if n_fired > 0:
                fire_counts += 1
            n_finite = sum(1 for d in ann.fired_divergence if math.isfinite(d))
            all_finite_div += n_finite

            chunk_agree_counts.append((len(ann.source_chunks), len(r["source_chunks"])))

            rec = {
                "index": r["index"],
                "latency": r["latency"],
                "n_src_tok": n,
                "n_tgt_tok": m,
                "commit": ann.commit_source_tok_idx,
                "fired_div": [d if math.isfinite(d) else None for d in ann.fired_divergence],
                "source_chunks": ann.source_chunks,
                "target_chunks": ann.target_chunks,
                "gpt4_source_chunks": r["source_chunks"],
                "gpt4_target_chunks": r["target_chunks"],
                "east_str": ann.east_str,
                "pearson_i_j": r_pearson,
                "dt_sec": dt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_sentence.append(rec)

            print(f"  [{k+1}/{len(kept)}] idx={r['index']} lat={r['latency']} "
                  f"n={n} m={m} chunks_ours={len(ann.source_chunks)} "
                  f"chunks_gpt4={len(r['source_chunks'])} pearson(i*/n, j/m)={r_pearson:.3f} "
                  f"fires={n_fired}/{m} dt={dt:.1f}s")

    total_dt = time.time() - t_annot
    print(f"\nAnnotated {len(per_sentence)}/{len(kept)} in {total_dt:.1f}s "
          f"({total_dt/max(len(per_sentence),1):.1f}s/sentence)")

    # Aggregate gate signals.
    def q(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        k = int(round(p * (len(xs) - 1)))
        return xs[k]

    ours_chunks = [c[0] for c in chunk_agree_counts]
    gpt4_chunks = [c[1] for c in chunk_agree_counts]

    summary = {
        "config": {
            "n_sentences_requested": args.n_sentences,
            "n_sentences_kept": len(kept),
            "n_sentences_annotated": len(per_sentence),
            "tau": args.tau,
            "criterion": args.criterion,
            "seed": args.seed,
            "model_path": args.model_path,
            "max_src_tokens": args.max_src_tokens,
        },
        "gate_signals": {
            "pearson_i_over_n_vs_j_over_m": {
                "min": min(correlations) if correlations else None,
                "median": q(correlations, 0.5),
                "max": max(correlations) if correlations else None,
                "n_gt_0.99": sum(1 for c in correlations if c > 0.99),
                "warn_if_median_gt_0.98": (q(correlations, 0.5) or 0) > 0.98,
            },
            "fire_fraction": fire_counts / max(len(per_sentence), 1),
            "chunks_per_sentence": {
                "ours_mean": (sum(ours_chunks) / max(len(ours_chunks), 1)) if ours_chunks else None,
                "gpt4_mean": (sum(gpt4_chunks) / max(len(gpt4_chunks), 1)) if gpt4_chunks else None,
            },
        },
        "env": {
            "device": device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": os.popen("git -C " + str(REPO_ROOT) + " rev-parse HEAD").read().strip(),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Sanity report (METHOD §8) ===")
    ps = summary["gate_signals"]["pearson_i_over_n_vs_j_over_m"]
    print(f"Pearson(i*/n, j/m): min={ps['min']:.3f} median={ps['median']:.3f} "
          f"max={ps['max']:.3f}  (>{ps['n_gt_0.99']}/{len(correlations)} above 0.99)")
    if ps["warn_if_median_gt_0.98"]:
        print("  WARN: median > 0.98 — positional degeneracy suspected.")
    print(f"Fire fraction: {summary['gate_signals']['fire_fraction']:.2%} of sentences had "
          f">=1 target token that committed before n.")
    print(f"Mean chunks/sentence: ours={summary['gate_signals']['chunks_per_sentence']['ours_mean']:.2f}  "
          f"gpt4={summary['gate_signals']['chunks_per_sentence']['gpt4_mean']:.2f}")

    print(f"\nWrote per-sentence traces to {annotations_path}")
    print(f"Wrote summary to {summary_path}")
    print("PHASE 1 SMOKE COMPLETE")


if __name__ == "__main__":
    main()
