"""
Plot BLEU vs AL for cond-A vs cond-B streaming eval results.

Mirrors EAST Fig. 4 style: one subplot per (backbone, language pair),
each with cond-A and cond-B curves across wait_k policies + check_argmax.

Reads results/phase2/extrinsic/full_stream_<policy>_<arm>_n<n>.json files
and produces figures/phase2/bleu_vs_al_<backbone>.pdf + .png.

Usage:
    python scripts/phase2_plot_bleu_al.py \
        --extrinsic_dir results/phase2/extrinsic \
        --backbones gemma-e2b \
        --n 10K \
        --output figures/phase2/bleu_vs_al_gemma-e2b.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_points(extrinsic_dir: Path, arm: str, tag: str, n_tag: str, policies):
    """Return sorted list of (AL, BLEU, policy_label)."""
    pts = []
    for policy_key, label in policies:
        f = extrinsic_dir / f"full_stream_{policy_key}_{arm}_{n_tag}{tag}.json"
        if not f.exists():
            print(f"  [MISS] {f.name}")
            continue
        d = json.loads(f.read_text())
        al = d["al_mean"]
        bleu = d["bleu"]
        pts.append((al, bleu, label))
    pts.sort(key=lambda p: p[0])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrinsic_dir", type=Path,
                    default=Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic"))
    ap.add_argument("--n_tag", default="n10k",
                    help="Filename suffix — 'n10k', 'n20k', etc. Matches --output_dir naming.")
    ap.add_argument("--backbone_tag", default="",
                    help="e.g. '_qwen35', '_e4b'. Empty for Gemma-4-E2B (default).")
    ap.add_argument("--title", default="Gemma-4-E2B (n=10K)")
    ap.add_argument("--output", type=Path,
                    default=Path("/g/data/ba39/dipankar/simt-tor-26/figures/phase2/bleu_vs_al_gemma-e2b.pdf"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Which policies to include. Order by expected AL.
    policies = [
        ("waitk1", "wait-1"),
        ("waitk3", "wait-3"),
        ("waitk5", "wait-5"),
        ("waitk7", "wait-7"),
        ("waitk9", "wait-9"),
        ("waitk11", "wait-11"),
        ("checkargmax", "check_argmax"),
    ]

    print(f"Collecting cond-A / cond-B points from {args.extrinsic_dir}")
    a_pts = collect_points(args.extrinsic_dir, "condA", args.backbone_tag, args.n_tag, policies)
    b_pts = collect_points(args.extrinsic_dir, "condB", args.backbone_tag, args.n_tag, policies)
    print(f"  cond-A: {len(a_pts)} points")
    print(f"  cond-B: {len(b_pts)} points")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if a_pts:
        xs, ys, labels = zip(*a_pts)
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=1.8,
                color="#1f77b4", label="cond-A (GPT-4 chunks)")
        for x, y, lbl in a_pts:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8, alpha=0.6)
    if b_pts:
        xs, ys, labels = zip(*b_pts)
        ax.plot(xs, ys, marker="s", markersize=7, linewidth=1.8,
                color="#d62728", label="cond-B (OT-annotator, ours)")
        for x, y, lbl in b_pts:
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.6)

    ax.set_xlabel("AL (Average Lagging, source words)", fontsize=11)
    ax.set_ylabel("BLEU", fontsize=11)
    ax.set_title(args.title + " — newstest2013 De→En, streaming", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png').name}")

    # Also print a paper-copy table.
    print("\n=== Data (LaTeX-ready) ===")
    print(f"{'Policy':<14} {'cond-A AL':<10} {'cond-A BLEU':<12} {'cond-B AL':<10} {'cond-B BLEU':<12} {'Δ BLEU':<8}")
    a_by = {lbl: (al, bleu) for al, bleu, lbl in a_pts}
    b_by = {lbl: (al, bleu) for al, bleu, lbl in b_pts}
    for _, label in policies:
        if label in a_by and label in b_by:
            aa, ab = a_by[label]
            ba, bb = b_by[label]
            print(f"{label:<14} {aa:<10.2f} {ab:<12.2f} {ba:<10.2f} {bb:<12.2f} {bb-ab:+.2f}")


if __name__ == "__main__":
    main()
