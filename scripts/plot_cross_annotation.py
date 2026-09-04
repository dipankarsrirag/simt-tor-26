"""Plot BLEU against Average Lagging for the cross-annotation experiments.

One panel per test set and direction, one line per system, all five latency
prompts plotted as measured (no filtering). Reads the summary CSV written by
scripts/collect_run_artifacts.py.

Usage:
    python scripts/plot_cross_annotation.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LATS = ["low", "low-medium", "medium", "medium-high", "high"]
PANELS = [("wmt15", "de-en"), ("wmt22", "de-en"), ("wmt22", "en-de"),
          ("wmt22", "ru-en"), ("wmt22", "en-ru"), ("iwslt17", "de-en"),
          ("iwslt17", "en-de"), ("iwslt17", "ar-en"), ("iwslt17", "en-ar"),
          ("iwslt15", "vi-en"), ("iwslt15", "en-vi")]
STYLE = {
    "gemma_2b_curated": ("#eb6834", "Gemma-2B, self-annotated", "2B self"),
    "gemma_4b_curated": ("#8e44ad", "Gemma-4B, self-annotated", "4B self"),
    "gemma_4b_from_2b_annot": ("#2a78d6", "Gemma-4B on the 2B's chunks", "4B<-2B"),
    "east_8b_from_2b_annot": ("#199e70", "Llama-3-8B on the 2B's chunks", "8B<-2B"),
}


def load(path):
    # Group the rows by system, then by panel, ordered by latency.
    data = {}
    for r in csv.DictReader(open(path)):
        key = (r["test_set"], r["direction"])
        data.setdefault(r["system"], {}).setdefault(key, {})[r["latency"]] = (
            float(r["al"]), float(r["bleu"]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/eval/summary_cross_annotation.csv")
    ap.add_argument("--out", default="figures/cross_annotation/bleu_vs_al.png")
    ap.add_argument("--baseline", default="gemma_2b_curated",
                    help="system the panel titles report deltas against")
    args = ap.parse_args()

    data = load(REPO / args.summary)
    systems = [s for s in STYLE if s in data]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), facecolor="white")
    axes = axes.ravel()

    for ax, key in zip(axes, PANELS):
        deltas = []
        for system in systems:
            pts = [data[system][key][l] for l in LATS if l in data[system].get(key, {})]
            if not pts:
                continue
            colour = STYLE[system][0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=colour,
                    lw=2, ms=5, markeredgecolor="white", markeredgewidth=1.2)
            if system != args.baseline and args.baseline in data:
                base = [data[args.baseline][key][l] for l in LATS
                        if l in data[args.baseline].get(key, {})]
                if len(base) == len(pts):
                    mean = (sum(p[1] for p in pts) - sum(b[1] for b in base)) / len(pts)
                    deltas.append(f"{STYLE[system][2]} {mean:+.1f}")
        title = f"{key[0].upper()} {key[1]}"
        if deltas:
            title += "\nvs 2B: " + "  ".join(deltas)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(True, color="#e9e8e3", lw=0.8)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Last cell carries the legend instead of a panel.
    axes[-1].axis("off")
    axes[-1].legend(handles=[plt.Line2D([], [], color=STYLE[s][0], marker="o",
                                        lw=2, label=STYLE[s][1]) for s in systems],
                    loc="center", fontsize=9.5, frameon=False)
    fig.suptitle("BLEU vs Average Lagging, all five latency prompts per system",
                 fontsize=12, x=0.01, ha="left")
    fig.supxlabel("AL (words behind the speaker)", fontsize=9)
    fig.supylabel("BLEU", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.96))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
