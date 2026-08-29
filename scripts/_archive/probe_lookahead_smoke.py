"""
Smoke test for the lookahead_k flag on annotate_pair.

Runs 3 de-en sentences at k ∈ {0, 1, 2, 3} and reports:
  * commit points (should differ across k)
  * divergence matrix row-0 mean (k=0 anchors at "converged to full", k>0
    anchors at "converged locally")
  * chunk count at fixed tau

Also runs a REAL byte-identity check: re-annotates the first row of
results/phase2/annot_ot_multi_de-en/matrices.jsonl (produced pre-refactor
at k=0) with the new code at k=0 and compares divergence values element-
wise. If they don't match to ~1e-3 (Sinkhorn iteration noise), the k=0
path is not equivalent to prior behaviour and the pilot should NOT be
trusted.

Usage: python scripts/probe_lookahead_smoke.py  (needs 1×GPU, ~5 min)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from src.annotator.annotate import annotate_pair, _chunks_from_commit, _enforce_monotone
from src.constants import PRIMARY_BACKBONE, REPO_ROOT


SENTENCES = [
    # de-en pairs — one short + one with front-loaded verb + one with verb-final
    dict(
        src_lang="German", tgt_lang="English",
        source="Die Katze schläft auf dem Sofa.",
        target="The cat sleeps on the sofa.",
    ),
    dict(
        src_lang="German", tgt_lang="English",
        source="Der Präsident hat gestern eine wichtige Rede gehalten.",
        target="The president gave an important speech yesterday.",
    ),
    dict(
        src_lang="German", tgt_lang="English",
        source="Er kündigte an, dass die Regierung neue Steuern einführen wird.",
        target="He announced that the government will introduce new taxes.",
    ),
]


def commit_from_matrix(matrix, tau: float, n: int) -> list[int]:
    """Same helper as phase1_tau_sweep.py — first i with div < tau, else n."""
    if not matrix or not matrix[0]:
        return []
    m = len(matrix[0])
    commit = [n] * m
    for j in range(m):
        for i in range(len(matrix)):
            d = matrix[i][j]
            if not math.isnan(d) and d < tau:
                commit[j] = i + 1  # 1-indexed prefix length
                break
    return _enforce_monotone(commit)


def run_one(model, tokenizer, sent, k: int, tau: float = 0.30):
    t0 = time.time()
    ann = annotate_pair(
        model=model, tokenizer=tokenizer,
        source=sent["source"], target=sent["target"],
        src_lang=sent["src_lang"], tgt_lang=sent["tgt_lang"],
        latency="medium",
        tau=0.0,                    # never fire during search — derive offline
        criterion_name="ot",
        return_full_matrix=True,
        prompt_mode="raw",
        lookahead_k=k,
    )
    dt = time.time() - t0
    matrix = ann.divergence_matrix
    n = ann.n_src_tok
    m = ann.n_tgt_tok
    commit = commit_from_matrix(matrix, tau, n)
    # Row-0 mean divergence over target tokens — sanity anchor
    row0 = matrix[0] if matrix else []
    row0_mean = sum(row0) / len(row0) if row0 else float("nan")
    # Last row (i=n) mean — should be 0 for k=0 (self-vs-self) and for k>0
    # too (last prefix commits against p_full which is itself for i=n only
    # via the trailing fallback; but by construction the trailing loop uses
    # p_full = P_pre[n], so div at position n = D(P_pre[n], P_full) = 0).
    last_row = matrix[n - 1] if matrix else []
    last_mean = sum(last_row) / len(last_row) if last_row else float("nan")
    return dict(
        k=k, n=n, m=m,
        commit=commit,
        row0_mean=row0_mean,
        last_row_mean=last_mean,
        chunks=len(set(commit)),
        dt=dt,
    )


def main():
    print(f"Loading {PRIMARY_BACKBONE} ...")
    device = "cuda"
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
    print("Model loaded. Running smoke ...")

    for i, sent in enumerate(SENTENCES):
        print(f"\n=== Sentence {i}: {sent['source'][:60]!r} ===")
        results = []
        for k in [0, 1, 2, 3]:
            r = run_one(model, tokenizer, sent, k)
            results.append(r)
            print(f"  k={k}: n={r['n']} m={r['m']} chunks(tau=0.30)={r['chunks']}  "
                  f"row0_mean={r['row0_mean']:.4f}  last_mean={r['last_row_mean']:.4f}  "
                  f"dt={r['dt']:.2f}s  commit={r['commit']}")
        # k=0 rerun for determinism sanity — should exactly match first k=0.
        r0b = run_one(model, tokenizer, sent, 0)
        det_ok = r0b["commit"] == results[0]["commit"]
        print(f"  k=0 rerun determinism: {'OK' if det_ok else 'MISMATCH'}")

    # === Byte-identity check against on-disk pre-refactor k=0 matrices ===
    # Re-annotate the first row of an existing matrices.jsonl and compare
    # element-wise. Sinkhorn is iterative; small differences (~1e-3) are
    # expected floating-point noise. Larger differences mean the refactor
    # broke the k=0 path.
    ondisk_path = REPO_ROOT / "results" / "phase2" / "annot_ot_multi_de-en" / "matrices.jsonl"
    if ondisk_path.exists():
        print("\n=== Byte-identity: on-disk k=0 matrices vs re-annotated ===")
        with open(ondisk_path) as f:
            ondisk = json.loads(f.readline())
        print(f"  Loaded idx={ondisk['index']} n={ondisk['n_src_tok']} m={ondisk['n_tgt_tok']}")
        pool_path = REPO_ROOT / "results" / "phase2" / \
            "multilingual_source_pool_v5_per_direction" / "de-en.json"
        with open(pool_path) as f:
            pool = json.load(f)
        row = next((r for r in pool if r.get("index") == ondisk["index"]), None)
        if row is None:
            print("  Pool row not found — skipping.")
        else:
            print(f"  Re-annotating source={row['source'][:80]!r} ...")
            ann = annotate_pair(
                model=model, tokenizer=tokenizer,
                source=row["source"], target=row["target"],
                src_lang=row["src_lang"], tgt_lang=row["tgt_lang"],
                latency=row.get("latency", "medium"),
                tau=0.0, criterion_name="ot",
                return_full_matrix=True,
                prompt_mode="raw",
                lookahead_k=0,
            )
            new_mat = ann.divergence_matrix
            old_mat = ondisk["matrix"]
            if len(new_mat) != len(old_mat) or len(new_mat[0]) != len(old_mat[0]):
                print(f"  SHAPE MISMATCH: new {len(new_mat)}x{len(new_mat[0])} "
                      f"vs old {len(old_mat)}x{len(old_mat[0])}")
            else:
                max_abs = 0.0
                mean_abs = 0.0
                count = 0
                for i in range(len(new_mat)):
                    for j in range(len(new_mat[0])):
                        a = new_mat[i][j]
                        b = old_mat[i][j]
                        if math.isnan(a) and math.isnan(b):
                            continue
                        d = abs(a - b)
                        max_abs = max(max_abs, d)
                        mean_abs += d
                        count += 1
                mean_abs /= max(count, 1)
                print(f"  Divergence delta: max={max_abs:.6f}  mean={mean_abs:.6f}  cells={count}")
                if max_abs < 1e-3:
                    print("  BYTE-IDENTITY: OK (within Sinkhorn noise floor)")
                elif max_abs < 1e-2:
                    print("  BYTE-IDENTITY: MARGINAL — Sinkhorn variance? investigate")
                else:
                    print("  BYTE-IDENTITY: FAIL — k=0 path diverges from prior behavior")
    else:
        print(f"\n=== On-disk matrices not found at {ondisk_path} — skipping identity check ===")

    print("\nDone.")


if __name__ == "__main__":
    main()
