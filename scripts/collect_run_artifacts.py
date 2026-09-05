"""Collect the artefacts of a finished run into the repo: cleaned PBS logs, a
loss curve per tag, a per-cell eval summary, and a manifest.

PBS logs are written with progress-bar redraws and per-sentence samples, which
are large and already captured elsewhere (the full outputs are in the eval
JSONs). This keeps the parts that describe the run.

Usage:
    python scripts/collect_run_artifacts.py --pbs_dir . --tags gemma_4b_curated
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LATS = ["low", "low-medium", "medium", "medium-high", "high"]
PROGRESS = re.compile(r"^\s*\d+%\|.*\|\s*\d+/\d+")
SAMPLE = re.compile(r"^\s*(\[\d+/\d+\] src=|hyp=|ref=|corpus AL so far)")
LOSS = re.compile(r"\{'(?:eval_)?loss':[^}]*\}")
WROTE = re.compile(r"wrote results/eval/[^/]+/(\S+)\.json")


def clean(path, keep_samples=3):
    # Keep the last segment of each carriage-return line, drop progress bars,
    # and keep only the first few per-sentence samples.
    out, blank, seen = [], 0, 0
    for raw in open(path, errors="replace"):
        line = raw.split("\r")[-1].rstrip()
        if PROGRESS.match(line) or line.startswith("Loading weights:"):
            continue
        stripped = line.strip()
        if SAMPLE.match(stripped):
            if stripped.startswith("["):
                seen += 1
            if seen > keep_samples:
                continue
        if not stripped:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out) + "\n"


def loss_curve(text):
    # Pull the trainer's log dicts out of a shard log.
    points = []
    for m in LOSS.finditer(text):
        try:
            points.append(ast.literal_eval(m.group(0)))
        except (ValueError, SyntaxError):
            continue
    return points


def find_logs(pbs_dir, patterns):
    # Job names differ between runs, so try each known pattern.
    for pat in patterns:
        hits = sorted(glob.glob(f"{pbs_dir}/{pat}"))
        if hits:
            return hits
    return []


def collect_train(pbs_dir, tag, num, out_dir):
    # Keep shards that logged training steps; count why the others stopped.
    out_dir.mkdir(parents=True, exist_ok=True)
    points, kept, dropped, reasons = [], 0, 0, {}
    for p in find_logs(pbs_dir, [f"simt-train{num}.o*", f"sft_{tag}.o*"]):
        text = clean(p)
        pts = loss_curve(text)
        if not pts:
            dropped += 1
            reason = ("cuda_out_of_memory" if "OutOfMemoryError" in text
                      else "disk_quota" if "Disk quota exceeded" in text
                      else "incomplete_checkpoint" if "trainer_state.json" in text
                      else "no_training_steps")
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept += 1
        points += pts
        (out_dir / f"train.o{p.rsplit('.o', 1)[-1]}.log").write_text(text)
    if points:
        points.sort(key=lambda d: float(d.get("epoch", 0)))
        with open(out_dir / "loss_curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss", "eval_loss",
                                              "learning_rate", "grad_norm"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(points)
    return dict(train_shards_with_steps=kept, train_shards_dropped=dropped,
                dropped_shard_reasons=reasons)


def collect_eval_logs(pbs_dir, tag, num, out_dir):
    # Name each log after the cell it evaluated; skip attempts with no output.
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0
    for p in find_logs(pbs_dir, [f"simt-eval{num}.o*", f"eval_{tag}.o*"]):
        text = clean(p)
        m = WROTE.search(text)
        if not m:
            dropped += 1
            continue
        cell = m.group(1).replace(f"_stream_{tag}_check_argmax", "")
        (out_dir / f"{cell}.o{p.rsplit('.o', 1)[-1]}.log").write_text(text)
        kept += 1
    return dict(eval_jobs_with_output=kept, eval_jobs_dropped=dropped)


def eval_rows(tag):
    # One row per eval cell: quality, lag and speed.
    rows = []
    for f in sorted(glob.glob(str(REPO / "results" / "eval" / tag / "*.json"))):
        d = json.load(open(f))
        stem = Path(f).stem.replace(f"_stream_{tag}_check_argmax", "")
        test_set, rest = stem.split("_", 1)
        latency, direction = rest.rsplit("_", 1)
        rows.append(dict(system=tag, test_set=test_set, direction=direction,
                         latency=latency, n_sentences=d["n_sentences"],
                         bleu=round(d["bleu"], 2), al=round(d["al_mean"], 2),
                         laal=round(d["laal_mean"], 2),
                         sec_per_sentence=round(d["sec_per_sentence"], 3)))
    return rows


def hub_eval_rows(repo_id, system, marker):
    # Read a system's eval cells straight from its Hub repo.
    from huggingface_hub import HfApi, hf_hub_download
    rows = []
    for f in HfApi().list_repo_files(repo_id):
        if marker not in f or not f.startswith("eval/"):
            continue
        try:
            d = json.load(open(hf_hub_download(repo_id, f)))
        except Exception:
            continue
        stem = Path(f).stem.replace(marker, "").rsplit("_n", 1)[0]
        stem = stem.replace("_low_medium_", "_low-medium_").replace("_medium_high_", "_medium-high_")
        test_set, rest = stem.split("_", 1)
        latency, direction = rest.rsplit("_", 1)
        rows.append(dict(system=system, test_set=test_set, direction=direction,
                         latency=latency, n_sentences=d["n_sentences"],
                         bleu=round(d["bleu"], 2), al=round(d["al_mean"], 2),
                         laal=round(d["laal_mean"], 2),
                         sec_per_sentence=round(d["sec_per_sentence"], 3)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbs_dir", default=".", help="where the simt-*.o* logs are")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--nums", nargs="+", required=True,
                    help="config number per tag, e.g. 05 06 07")
    ap.add_argument("--hub_baseline", default=None,
                    help="repo_id:system:filename_marker for a system whose eval "
                         "JSONs live on the Hub, added to the summary as-is")
    ap.add_argument("--summary", default="results/eval/summary_cross_annotation.csv")
    args = ap.parse_args()

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    all_rows = []
    for tag, num in zip(args.tags, args.nums):
        train_dir = REPO / "logs" / "train" / tag
        stats = collect_train(args.pbs_dir, tag, num, train_dir)
        stats.update(collect_eval_logs(args.pbs_dir, tag, num,
                                       REPO / "logs" / "eval" / tag))
        build_dir = REPO / "logs" / "sft_dataset" / tag
        build_dir.mkdir(parents=True, exist_ok=True)
        for p in find_logs(args.pbs_dir, [f"simt-sft{num}-build.o*", f"build_{tag}.o*", "build_m90k.o*"]):
            (build_dir / f"build.o{p.rsplit('.o', 1)[-1]}.log").write_text(clean(p))

        rows = eval_rows(tag)
        all_rows += rows
        dataset = REPO / "results" / "sft_dataset" / tag / "sft_dataset.json"
        manifest = dict(tag=tag, config=f"configs/{num}_{tag}.yaml", git_sha=sha,
                        cluster="UNSW Katana (PBS)",
                        sft_rows=len(json.load(open(dataset))) if dataset.exists() else None,
                        eval_cells=len(rows), **stats)
        (train_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(tag, stats, len(rows), "eval cells")

    systems = list(args.tags)
    if args.hub_baseline:
        repo_id, system, marker = args.hub_baseline.split(":")
        base = hub_eval_rows(repo_id, system, marker)
        all_rows = base + all_rows
        systems = [system] + systems
        print(system, len(base), "eval cells from", repo_id)

    if all_rows:
        order = {t: i for i, t in enumerate(systems)}
        all_rows.sort(key=lambda r: (order[r["system"]], r["test_set"],
                                     r["direction"], LATS.index(r["latency"])))
        out = REPO / args.summary
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print("wrote", out, len(all_rows), "rows")


if __name__ == "__main__":
    main()
