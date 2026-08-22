"""Plot BLEU vs AL for all 8 directions × 5 conditions, styled like EAST Fig 4.

8 subplots (one per direction): de-en, en-de, ar-en, en-ar, ru-en, en-ru, vi-en, en-vi.
Each shows a BLEU-vs-AL curve for:
  - ctrl (raw OT)          — our starting point
  - merged (<2 word merge)  — EAST §3.1 rule
  - merged3 (<=3 word merge) — aggressive merge
  - E4B on raw OT           — scaling test (2B → 4B)
  - cond-A (GPT-4 chunks)   — reference baseline (4 dirs only)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

EXTR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic")
LAT = ["low", "low_medium", "medium", "medium_high", "high"]
DIR = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]

CONDITIONS = [
    # (prefix,                                    label,             marker, color)
    ("flores_stream_v6bctrl_checkargmax",        "ctrl (raw OT)",    "o",    "#1f77b4"),
    ("flores_stream_v6bmerged_checkargmax",      "merged (<2)",      "s",    "#2ca02c"),
    ("flores_stream_v6bmerged3_checkargmax",     "merged3 (<=3)",    "^",    "#d62728"),
    ("flores_stream_v6be4b_checkargmax",         "E4B (raw OT)",     "D",    "#9467bd"),
    ("flores_stream_v6bcondA_checkargmax",       "cond-A (GPT-4)",   "P",    "#ff7f0e"),
]


def load(prefix, lat, d):
    fp = EXTR / f"{prefix}_{lat}_{d}_n50.json"
    if not fp.exists():
        return None
    dd = json.loads(fp.read_text())
    return (dd.get("al_mean"), dd["bleu"])


def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, d in enumerate(DIR):
        ax = axes[i]
        for prefix, label, marker, color in CONDITIONS:
            xs, ys = [], []
            for lat in LAT:
                r = load(prefix, lat, d)
                if r is not None and r[0] is not None:
                    xs.append(r[0])
                    ys.append(r[1])
            if xs:
                # Sort by AL for line ordering
                order = sorted(range(len(xs)), key=lambda k: xs[k])
                xs = [xs[k] for k in order]
                ys = [ys[k] for k in order]
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=1.5, markersize=7, alpha=0.85)
        ax.set_title(f"({chr(ord('a') + i)}) {d}", fontsize=13)
        ax.set_xlabel("AL", fontsize=11)
        ax.set_ylabel("SacreBLEU", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(
        "BLEU vs AL on FLORES-200 devtest (N=50, streaming check_argmax, Gemma-4-E2B-it "
        "unless noted)\\n5 latencies per line (low, low-med, medium, med-high, high)",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()

    out_pdf = Path("/g/data/ba39/dipankar/simt-tor-26/figures/phase2/bleu_vs_al_all_conditions_flores_n50.pdf")
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}", flush=True)
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
