"""Plot BLEU-vs-AL and COMET-vs-AL for all 8 directions × N conditions.

Renders two figures (PNG only):
  figures/phase2/bleu_vs_al_all_conditions_flores_n50.png
  figures/phase2/comet_vs_al_all_conditions_flores_n50.png

8 subplots per figure (one per direction). Each shows a curve per condition:
  - ctrl (raw OT)          — our starting point
  - merged (<2 word merge) — EAST §3.1 rule
  - merged3 (<=3 word merge) — aggressive merge
  - merged3 rebucket        — 2026-08-22 (cc, sw) latency rule
  - merged3 rb+aug          — rebucket + latency-augmentation
  - E4B on raw OT           — scaling test (2B → 4B)
  - cond-A (GPT-4 chunks)   — reference baseline (4 dirs only)

COMET is loaded from `results/phase2/extrinsic/comet_scores_{tag}.json`
(produced by `phase2_score_comet.py`). Conditions without a comet_scores_*
file are silently dropped from the COMET plot.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

EXTR = Path("/g/data/ba39/dipankar/simt-tor-26/results/phase2/extrinsic")
FIG = Path("/g/data/ba39/dipankar/simt-tor-26/figures/phase2")
LAT = ["low", "low_medium", "medium", "medium_high", "high"]
DIR = ["de-en", "en-de", "ar-en", "en-ar", "ru-en", "en-ru", "vi-en", "en-vi"]

# Default: N=50 sanity outputs. Override via --n_suffix for large-N (e.g. n1012).
N_SUFFIX = "n50"

# (prefix,                                  comet_tag,       label,                 marker, color)
CONDITIONS = [
    ("flores_stream_v6bctrl_checkargmax",       "v6bctrl",       "ctrl (raw OT)",        "o", "#1f77b4"),
    ("flores_stream_v6bm3rbfw_checkargmax",     "v6bm3rbfw",     "rb+fw (prior best)",   "v", "#e377c2"),
    ("flores_stream_v6bv2bal_checkargmax",      "v6bv2bal",      "v2bal (ours, new)",    "*", "#2ca02c"),
    ("flores_stream_v6bcondA_checkargmax",      "v6bcondA",      "cond-A (GPT-4)",       "P", "#ff7f0e"),
]


def load_bleu_al(prefix, lat, d):
    fp = EXTR / f"{prefix}_{lat}_{d}_{N_SUFFIX}.json"
    if not fp.exists():
        return None
    dd = json.loads(fp.read_text())
    al = dd.get("al_mean")
    if al is None:
        return None
    return al, dd["bleu"]


_COMET_CACHE = {}


def load_comet_mean(comet_tag, lat, d):
    # Try N-suffixed file first (e.g. comet_scores_<tag>_n1012.json), fall back
    # to the un-suffixed (implicit n50) file.
    cache_key = (comet_tag, N_SUFFIX)
    if cache_key not in _COMET_CACHE:
        suffixed = EXTR / f"comet_scores_{comet_tag}_{N_SUFFIX}.json"
        default = EXTR / f"comet_scores_{comet_tag}.json"
        if suffixed.exists():
            _COMET_CACHE[cache_key] = json.loads(suffixed.read_text())
        elif N_SUFFIX == "n50" and default.exists():
            _COMET_CACHE[cache_key] = json.loads(default.read_text())
        else:
            _COMET_CACHE[cache_key] = None
    dat = _COMET_CACHE[cache_key]
    if dat is None:
        return None
    entry = dat.get(f"{lat}__{d}")
    return entry.get("comet_mean") if entry else None


def _render(y_loader, y_label, suptitle, out_png):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, d in enumerate(DIR):
        ax = axes[i]
        for prefix, comet_tag, label, marker, color in CONDITIONS:
            xs, ys = [], []
            for lat in LAT:
                ba = load_bleu_al(prefix, lat, d)
                if ba is None:
                    continue
                al = ba[0]
                y = y_loader(prefix, comet_tag, lat, d, ba)
                if y is None:
                    continue
                xs.append(al)
                ys.append(y)
            if xs:
                order = sorted(range(len(xs)), key=lambda k: xs[k])
                xs = [xs[k] for k in order]
                ys = [ys[k] for k in order]
                ax.plot(xs, ys, marker=marker, color=color, label=label,
                        linewidth=1.5, markersize=7, alpha=0.85)
        ax.set_title(f"({chr(ord('a') + i)}) {d}", fontsize=13)
        ax.set_xlabel("AL", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(suptitle, fontsize=12, y=1.005)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)


def main():
    global N_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_suffix", default="n50",
                    help="Output-file N suffix, e.g. 'n50' (default) or 'n1012' "
                         "(full FLORES devtest). Controls both the input JSON "
                         "filename pattern and the output PNG filename.")
    args = ap.parse_args()
    N_SUFFIX = args.n_suffix

    # BLEU
    _render(
        y_loader=lambda prefix, tag, lat, d, ba: ba[1],
        y_label="SacreBLEU",
        suptitle=(
            f"BLEU vs AL on FLORES-200 devtest (N={N_SUFFIX[1:]}, streaming check_argmax, "
            "Gemma-4-E2B-it unless noted)\n"
            "5 latencies per line (low, low-med, medium, med-high, high)"
        ),
        out_png=FIG / f"bleu_vs_al_all_conditions_flores_{N_SUFFIX}.png",
    )
    # COMET (may or may not be computed for large-N yet)
    _render(
        y_loader=lambda prefix, tag, lat, d, ba: load_comet_mean(tag, lat, d),
        y_label="COMET (wmt22-comet-da)",
        suptitle=(
            f"COMET vs AL on FLORES-200 devtest (N={N_SUFFIX[1:]}, streaming check_argmax, "
            "Gemma-4-E2B-it unless noted)\n"
            "5 latencies per line (low, low-med, medium, med-high, high)"
        ),
        out_png=FIG / f"comet_vs_al_all_conditions_flores_{N_SUFFIX}.png",
    )


if __name__ == "__main__":
    main()
