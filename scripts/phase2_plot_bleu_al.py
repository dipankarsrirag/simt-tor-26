"""
Plot BLEU vs AL for the paper's method-family figures.

Two figures per method-family split (see docs/05-phase2 "Cross-paper
comparability protocol", 2026-08-18 late):

  Fig. 1 — vs non-LLM SiMT (encoder-decoder tradition) on WMT15 De→En / AL.
           Competitors: ITST, SM²/SimulMask, HMT, wait-k baseline (verbatim
           from published tables). Our method (OT-SFT) + WaitK-SFT baseline
           computed here. Dashed reference line for EAST-Stage-I/8B/660K.

  Fig. 2 — vs LLM SiMT on WMT22 De→En / LAAL. Competitors: EAST, Simul-LLM,
           TransLLaMa, SimulPL, ConversationalSiMT (verbatim from tables).
           Our method + WaitK-SFT (this session) plus reference lines.

Our-method data source: results/phase2/extrinsic/full_stream_<policy>_n<n>.json
(single arm — OT-SFT; cond-A and Cond-C were both removed 2026-08-18 late).

Competitor numbers must be hand-entered from published tables — this
script hosts them as constants at the top for full transparency. Update
citations in RELATEDWORKS.md when changing them.

Usage:
    python scripts/phase2_plot_bleu_al.py \
        --extrinsic_dir results/phase2/extrinsic \
        --fig 1 \
        --output figures/phase2/fig1_vs_non_llm.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ─── Competitor numbers from published tables (verbatim) ──────────────
# EDIT WITH CITATIONS. All AL/LAAL in source words. BLEU is SacreBLEU-13a
# where reported; ITST + HMT use Moses multi-bleu.perl (see §Experiments
# "Cross-paper comparability").

# Fig. 1 — WMT15 De→En / AL / non-LLM SiMT competitors
FIG1_COMPETITORS = {
    # Method: list of (AL, BLEU) points; comment lists source table.
    "ITST":   [],  # TODO: fill from ITST Table X, WMT15 De→En
    "SM²-Bi": [],  # TODO: fill from EAST Fig. 3 (they include this)
    "HMT":    [],  # TODO: fill from HMT paper
    "wait-k baseline (encoder-decoder)": [],  # TODO: standard reference
}

# Fig. 2 — WMT22 De→En / LAAL / LLM SiMT competitors
FIG2_COMPETITORS = {
    "EAST (8B, 660K)": [  # EAST Table 3, WMT22 De→En, low/med/high latency
        # (LAAL, BLEU) — note EAST reports AL; LAAL is typically 0.2-1.5 higher
        # Values below are from EAST Table 3 as AL; convert to LAAL when known.
        (2.59, 29.87),
        (3.42, 31.08),
        (5.87, 32.38),
    ],
    "Simul-LLM": [],   # TODO: from Agostinelli et al. ACL 2024
    "TransLLaMa": [],  # TODO: from Koshkin et al. Findings EMNLP 2024
    "SimulPL": [],     # TODO: from SimulPL paper
    "Conversational SimulMT": [],  # TODO: from Wang et al. 2024
    "Llama3-MOMT w/ wait-k": [  # EAST Table 3
        (2.70, 26.50),
        (3.63, 27.60),
        (5.44, 28.95),
    ],
}


# ─── Our data collection ──────────────────────────────────────────────

# Single-arm plot: our OT-SFT method. Competitors come from the FIG*_COMPETITORS
# dicts above (published verbatim). Cond-A and Cond-C were removed 2026-08-18.
OUR_LABEL = "OT-SFT (ours)"
OUR_COLOR = "#d62728"
OUR_MARKER = "s"

POLICIES = [
    ("waitk1",       "wait-1"),
    ("waitk3",       "wait-3"),
    ("waitk5",       "wait-5"),
    ("waitk7",       "wait-7"),
    ("waitk9",       "wait-9"),
    ("waitk11",      "wait-11"),
    ("checkargmax",  "check_argmax"),
]


def collect_points(extrinsic_dir: Path, tag: str, n_tag: str, policies, metric_key="al_mean"):
    """Return sorted list of (metric, BLEU, policy_label). metric_key is
    'al_mean' for AL (Fig. 1) or 'laal_mean' for LAAL (Fig. 2). Reads
    full_stream_<policy>_<n_tag><tag>.json under the single-arm naming."""
    pts = []
    for policy_key, label in policies:
        f = extrinsic_dir / f"full_stream_{policy_key}_{n_tag}{tag}.json"
        if not f.exists():
            print(f"  [MISS] {f.name}")
            continue
        d = json.loads(f.read_text())
        m = d.get(metric_key)
        bleu = d.get("bleu")
        if m is None:
            print(f"  [MISS] {f.name} has no {metric_key} (pre-LAAL era; rerun to populate)")
            continue
        pts.append((m, bleu, label))
    pts.sort(key=lambda p: p[0])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrinsic_dir", type=Path,
                    default=Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic"))
    ap.add_argument("--n_tag", default="n10k",
                    help="Filename suffix — 'n10k', 'n20k', etc.")
    ap.add_argument("--backbone_tag", default="",
                    help="e.g. '_qwen35', '_e4b'. Empty for Gemma-4-E2B.")
    ap.add_argument("--fig", type=int, choices=[1, 2], required=True,
                    help="1 = vs non-LLM (WMT15 De→En/AL); 2 = vs LLM (WMT22 De→En/LAAL).")
    ap.add_argument("--title", default=None,
                    help="Override figure title. Default: derived from --fig.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.fig == 1:
        metric_key = "al_mean"
        xlabel = "AL (Average Lagging, source words) [Ma et al. 2019]"
        title = args.title or "Fig. 1 — vs non-LLM SiMT on WMT15 De→En"
        competitors = FIG1_COMPETITORS
    else:
        metric_key = "laal_mean"
        xlabel = "LAAL (Length-Adaptive AL) [Papi et al. 2022]"
        title = args.title or "Fig. 2 — vs LLM SiMT on WMT22 De→En"
        competitors = FIG2_COMPETITORS

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot competitors (published verbatim; open markers to signal source).
    for name, points in competitors.items():
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.4, alpha=0.7,
                linestyle="--", label=name, markerfacecolor="white")

    # Plot our OT-SFT arm (solid marker; single-arm since 2026-08-18).
    print(f"Collecting OT-SFT points from {args.extrinsic_dir} (metric={metric_key})")
    pts = collect_points(args.extrinsic_dir, args.backbone_tag, args.n_tag, POLICIES, metric_key=metric_key)
    print(f"  OT-SFT: {len(pts)} points")
    if pts:
        xs, ys, labels = zip(*pts)
        ax.plot(xs, ys, marker=OUR_MARKER, markersize=8, linewidth=2.0,
                color=OUR_COLOR, label=OUR_LABEL)
        for x, y, lbl in pts:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.6)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("SacreBLEU (13a)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png').name}")

    # Print a paper-copy table.
    print("\n=== OT-SFT data (LaTeX-ready) ===")
    for m, bleu, lbl in pts:
        print(f"  {lbl:<14} {metric_key}={m:.2f}  BLEU={bleu:.2f}")


if __name__ == "__main__":
    main()
