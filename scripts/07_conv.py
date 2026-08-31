"""
Stage 2 alternative — Conversational SiMT chunking (Wang et al. 2024).

Reference: Wang, Vu, Shareghi & Haffari, "Conversational SimulMT:
Efficient Simultaneous Translation with Large Language Models",
arXiv 2402.10552, §2.

Recipe (paper §2, verbatim):
    "we employ fastalign to obtain word alignment between source and
     target tokens ... We then segment the monotonic dependency graph
     and convert these segments into READ / WRITE pairs, representing
     the meta trajectory of the oracle policy with minimum latency."

Formally: for target-token indices j = 1..m with alignment sets
    aⱼ = { source-token indices aligned to target token yⱼ }
the READ/WRITE pairs are
    Rⱼ = { xᵢ | i ∈ aⱼ ∖ aⱼ₋₁ }   (new source tokens revealed at step j)
    Wⱼ = { yⱼ }                     (single target token per WRITE)
with monotonicity enforced (aⱼ ≥ max(aⱼ₋₁)) so Rⱼ contains a suffix
of already-read source. Adjacent Wⱼ's whose Rⱼ is empty are merged.

Deviations from Wang, all logged in `LOG.md` (2026-08-31):

  - **Alignment tool.** We use `awesome-align` (mBERT-based, pip-
    installable) instead of `fastalign`. Rationale: fastalign requires
    a copyq C++ build; awesome-align is higher-precision and matches
    the "adopt theirs and log the deviation" clause of
    `docs/followup-experiments.md` §Pre-registered hyperparameters.

  - **Inference-time policy.** Wang uses RALCP (Relaxed Agreement LCP
    over multiple hypotheses) for WRITE termination. We simplify to
    greedy-per-chunk (read n source tokens, generate until model emits
    `<|end-of-write|>`) so all three baselines (OT / wait-k / conv-simt)
    share the same streaming inference loop in
    `src/eval/extrinsic.py::stream_translate`.

STATUS: NOT YET IMPLEMENTED.

Remaining blockers (must resolve before this runs):

  [1] `awesome-align` install + `bert-base-multilingual-cased` weights.
      Queued as copyq job 177879264 (`jobs/download/
      install_awesome_align_and_mbert.pbs`). Post-job, verify:
          python -c 'import awesome_align'
          ls /g/data/po67/dipankar/models/mBERT/config.json

  [2] Per-latency training strategy — Wang trains one model and sweeps
      `n` at inference; we need per-row latency labels for EAST-format
      training. Options (see `LOG.md` [DECISION] 2026-08-31 Conv-SiMT
      recipe):
        (a) Single training run, label all rows `medium`, skip low/high.
        (b) Train 3× (one per latency) — merge alignment-derived chunks
            to target n≈3 (low) / 5 (medium) / 7 (high) source tokens.
        (c) Per-row latency via `latency_from_chunk_stats` (matches OT
            arm; no explicit control over the per-latency chunk-size).
      Recommend (b). Awaiting user decision.

Usage (once implemented):
    bin/07_conv --config configs/04_east_8b_conv.yaml \\
                --output results/sft_dataset/east_8b_conv/sft_dataset.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config


ALIGN_MODEL = "/g/data/po67/dipankar/models/mBERT"   # set once job 177879264 completes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["annotate"]["criterion"] != "conv-simt":
        sys.exit(f"config criterion is {cfg['annotate']['criterion']!r}, expected 'conv-simt'")

    raise NotImplementedError(
        "conv-simt is not yet implemented. See the module docstring for the "
        "remaining blockers: (1) awesome-align install (queued as 177879264) "
        "and (2) per-latency training strategy (a/b/c) awaiting user decision. "
        f"Config validated OK: tag={cfg['tag']}, "
        f"latency_bins={cfg['annotate']['latency_bins']}."
    )


if __name__ == "__main__":
    main()
