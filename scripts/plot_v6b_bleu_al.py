"""Regenerate the 5 clean-eval BLEU-vs-AL plots for the v6b paper.

Outputs (PNG only, dropped in figures/phase2/):
  bleu_al_flores.png         — 2x4 grid, all 8 directions
  bleu_al_wmt15_de-en.png    — single subplot
  bleu_al_wmt22.png          — 1x4 (de/ru bidirectional)
  bleu_al_iwslt17.png        — 1x4 (de/ar bidirectional)
  bleu_al_iwslt15_vi.png     — 1x2 (vi bidirectional; CondA skipped)

Models:
  CondA  = v6bcondA         red    ^
  CondB  = v6bv2balv3       blue   D   (de/ru only on non-flores; hidden on ar/vi)
  Ours   = v6bv2balv3htgt   purple s

Serif fonts. x-tick 2 units. y-tick 4 units.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EXTR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic")
FIG = Path("/g/data/ba39/dipankar/simt-tor-26/figures/phase2")
FIG.mkdir(parents=True, exist_ok=True)

LAT = ["low", "low_medium", "medium", "medium_high", "high"]

MODELS = [
    ("v6bcondA",       "CondA",   "#c0392b", "^"),
    ("v6bv2balv3",     "CondB",   "#2c6fbb", "D"),
    ("v6bv2balv3htgt", "Ours",    "#7d3c98", "s"),
    ("east8b",         "EAST-8B", "#2e8b57", "o"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Nimbus Roman", "Times New Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def load_curve(model_tag: str, ds_prefix: str, direction: str, n_suffix: str,
               hide_condB_on_ar_vi: bool = False) -> tuple[list[float], list[float]]:
    """Return (xs=AL, ys=BLEU) sorted by AL."""
    if hide_condB_on_ar_vi and model_tag == "v6bv2balv3" and direction.split("-")[0] in {"ar", "vi"} \
            and direction.split("-")[1] in {"ar", "vi", "en"} and ("ar" in direction or "vi" in direction):
        return [], []
    xs, ys = [], []
    for lat in LAT:
        fp = EXTR / f"{ds_prefix}_stream_{model_tag}_checkargmax_{lat}_{direction}_{n_suffix}.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        al = d.get("al_mean")
        bleu = d.get("bleu")
        if al is None or bleu is None:
            continue
        xs.append(al)
        ys.append(bleu)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def style_axes(ax, xmax_hint: float | None = None):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_xlabel("AL")
    ax.set_ylabel("SacreBLEU")


def plot_one(ax, ds_prefix: str, direction: str, n_suffix: str,
             hide_condB: bool = False, show_legend: bool = True):
    plotted = False
    for tag, label, color, marker in MODELS:
        if hide_condB and tag == "v6bv2balv3":
            continue
        xs, ys = load_curve(tag, ds_prefix, direction, n_suffix)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linewidth=1.8, markersize=8, alpha=0.9)
        plotted = True
    style_axes(ax)
    if plotted and show_legend:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    return plotted


def plot_flores():
    directions = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, d in enumerate(directions):
        ax = axes[i]
        # CondB shown only on de/ru
        hide_condB = d.split("-")[0] in {"ar", "vi"} or d.split("-")[1] in {"ar", "vi"}
        plotted = plot_one(ax, "flores", d, "n1012",
                           hide_condB=hide_condB, show_legend=(i == 0))
        ax.set_title(f"({chr(ord('a')+i)}) {d}")
    fig.suptitle("BLEU vs AL on FLORES-200 devtest (N=1012)  —  contaminated eval set", y=1.005)
    fig.tight_layout()
    out = FIG / "bleu_al_flores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_wmt15():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_one(ax, "wmt15", "de-en", "n2169")
    ax.set_title("WMT15 newstest2015 De→En")
    fig.tight_layout()
    out = FIG / "bleu_al_wmt15_de-en.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_wmt22():
    dirs = [("de-en", "n1979"), ("en-de", "n1904"), ("ru-en", "n2016"), ("en-ru", "n2037")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (d, n) in enumerate(dirs):
        ax = axes[i]
        plot_one(ax, "wmt22", d, n, show_legend=(i == 0))
        ax.set_title(f"({chr(ord('a')+i)}) WMT22 {d}")
        ax.set_ylim(15, 40)
        ax.yaxis.set_major_locator(MultipleLocator(5))
    fig.tight_layout()
    out = FIG / "bleu_al_wmt22.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_iwslt17():
    # de-en / en-de: n1138; ar-en / en-ar: n1460
    dirs = [("de-en", "n1138"), ("en-de", "n1138"), ("ar-en", "n1460"), ("en-ar", "n1460")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, (d, n) in enumerate(dirs):
        ax = axes[i]
        # ar/en-ar: hide CondB
        hide_condB = "ar" in d
        plot_one(ax, "iwslt17", d, n, hide_condB=hide_condB, show_legend=(i == 0))
        ax.set_title(f"({chr(ord('a')+i)}) IWSLT17 {d}")
    fig.tight_layout()
    out = FIG / "bleu_al_iwslt17.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_iwslt15_vi():
    dirs = [("vi-en", "n1268"), ("en-vi", "n1268")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (d, n) in enumerate(dirs):
        ax = axes[i]
        # vi: hide CondB (CondA has no vi coverage anyway)
        hide_condB = True
        plot_one(ax, "iwslt15", d, n, hide_condB=hide_condB, show_legend=(i == 0))
        ax.set_title(f"({chr(ord('a')+i)}) IWSLT15 {d}")
    fig.tight_layout()
    out = FIG / "bleu_al_iwslt15_vi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_flores()
    plot_wmt15()
    plot_wmt22()
    plot_iwslt17()
    plot_iwslt15_vi()
