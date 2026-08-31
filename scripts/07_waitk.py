"""
Stage 2 alternative — procedural wait-k chunking (no LM forward pass).

Reads the per-direction source-pool JSONs produced by Stage 1
(`06_build_source_pool.py`), applies the wait-k policy at each configured
latency, and writes an SFT-ready dataset JSON directly. Stage 3
(`08_build_sft_dataset.py`) is skipped for wait-k configs — this script
produces the same output schema.

Wait-k policy (matches `src/eval/extrinsic.py` streaming policy=`wait_k`):
    commit an EOR every k source WORDS read. `k` is set per latency from
    the config's `annotate.wait_k_per_latency` mapping.

For each source-target pair × each configured latency, one training row is
produced. Source is split at whitespace (or per-character for CJK); target
is proportionally allocated across source-triggered chunks.

Output schema per row (matches what `08_build_sft_dataset.py` emits):
    {
      index, source, target, src_lang, tgt_lang, latency,
      source_chunks, target_chunks,
      source_chunk_ids, target_chunk_ids,
      _annotator_meta: {criterion: "wait-k", k: <int>},
    }

Usage:
    bin/07_waitk --config configs/03_east_8b_waitk.yaml \\
                 --output results/sft_dataset/east_8b_waitk/sft_dataset.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.annotator.annotate import _is_cjk_lang
from src.annotator.east_format import LATENCY_NL
from src.config import REPO_ROOT, load_config, resolve_backbone_path


def split_words(text: str, lang: str) -> List[str]:
    """Whitespace split for non-CJK; per-character for CJK."""
    if _is_cjk_lang(lang):
        return [c for c in text.replace(" ", "")]
    return text.split()


def tokenize_by_words(tokenizer, text: str, lang: str) -> Tuple[List[int], List[List[int]]]:
    """Return (full_ids, per_word_spans). Matches
    src/eval/extrinsic.py::tokenize_source_by_words semantics: word[0] has
    no leading space; word[i>0] gets a single leading space; concatenating
    per-word spans equals tokenizing the whole string.

    Falls back to offset_mapping attribution when naive per-word tokenization
    doesn't equal the full-string tokenization (rare boundary-merge cases).
    """
    is_cjk = _is_cjk_lang(lang)
    if is_cjk:
        clean = text.replace(" ", "")
        chars = list(clean)
        spans = [tokenizer(c, add_special_tokens=False).input_ids for c in chars]
        naive = [t for s in spans for t in s]
        full = tokenizer(clean, add_special_tokens=False).input_ids
        if naive == full:
            return full, spans
        # Fallback via offset_mapping
        enc = tokenizer(clean, add_special_tokens=False, return_offsets_mapping=True)
        full = enc.input_ids
        offsets = enc.offset_mapping
        spans = [[] for _ in chars]
        for tid, (a, _b) in zip(full, offsets):
            ci = a if a < len(chars) else max(0, len(chars) - 1)
            spans[min(ci, len(chars) - 1)].append(tid)
        return full, spans

    words = text.split()
    spans = []
    for wi, w in enumerate(words):
        prefix = w if wi == 0 else " " + w
        spans.append(tokenizer(prefix, add_special_tokens=False).input_ids)
    naive = [t for s in spans for t in s]
    full = tokenizer(text, add_special_tokens=False).input_ids
    if naive == full:
        return full, spans
    # Fallback via offset_mapping
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    full = enc.input_ids
    offsets = enc.offset_mapping
    word_of_char = [-1] * len(text)
    ci = 0
    for wi, w in enumerate(words):
        while ci < len(text) and text[ci].isspace():
            ci += 1
        for _ in range(len(w)):
            if ci < len(text):
                word_of_char[ci] = wi
                ci += 1
    spans = [[] for _ in words]
    for tid, (a, _b) in zip(full, offsets):
        wi = word_of_char[a] if a < len(text) else -1
        if wi >= 0:
            spans[wi].append(tid)
    return full, spans


def waitk_chunks(n_src_words: int, n_tgt_words: int, k: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (src_start, src_end, tgt_start, tgt_end) chunk spans.

    Policy: commit an EOR every k source words. Target words are proportionally
    allocated across the resulting source-triggered chunks. Any leftover source
    at the tail is absorbed into the last chunk.

    Guarantees:
      - concat of source spans covers [0, n_src_words) exactly
      - concat of target spans covers [0, n_tgt_words) exactly
      - each chunk has at least 1 source word and 1 target word (unless the
        pair itself has an empty side)
    """
    if n_src_words <= 0 or n_tgt_words <= 0:
        return [(0, n_src_words, 0, n_tgt_words)]

    # Source-side commit points: after k, 2k, 3k, ... source words. Cap at n.
    n_chunks = max(1, math.ceil(n_src_words / k))
    src_bounds = [0]
    for i in range(1, n_chunks):
        src_bounds.append(min(i * k, n_src_words))
    src_bounds.append(n_src_words)  # absorb any tail into last chunk

    # Target proportional split — align j to source-word ratio.
    tgt_bounds = [0]
    for i in range(1, n_chunks):
        tgt_bounds.append(round(i * n_tgt_words / n_chunks))
    tgt_bounds.append(n_tgt_words)

    # Ensure each chunk has >= 1 target word (rare — very short target sentences)
    # by collapsing empty target chunks INTO the preceding chunk.
    spans = []
    for i in range(n_chunks):
        ss, se = src_bounds[i], src_bounds[i + 1]
        ts, te = tgt_bounds[i], tgt_bounds[i + 1]
        if te <= ts and spans:
            prev = spans[-1]
            spans[-1] = (prev[0], se, prev[2], te)
        else:
            spans.append((ss, se, ts, te))
    return spans


def build_row(row: Dict, tokenizer, k: int, latency: str,
              max_src_tokens: int) -> Dict | None:
    """Produce one SFT training row from (row, k, latency)."""
    src_lang = row["src_lang"]
    tgt_lang = row["tgt_lang"]
    src = row["source"].strip()
    tgt = row["target"].strip()

    src_full_ids, src_word_spans = tokenize_by_words(tokenizer, src, src_lang)
    if len(src_full_ids) > max_src_tokens:
        return None
    tgt_full_ids, tgt_word_spans = tokenize_by_words(tokenizer, tgt, tgt_lang)

    n_src_w = len(src_word_spans)
    n_tgt_w = len(tgt_word_spans)
    if n_src_w == 0 or n_tgt_w == 0:
        return None

    src_words = split_words(src, src_lang)
    tgt_words = split_words(tgt, tgt_lang)
    src_sep = "" if _is_cjk_lang(src_lang) else " "
    tgt_sep = "" if _is_cjk_lang(tgt_lang) else " "

    spans = waitk_chunks(n_src_w, n_tgt_w, k)
    source_chunks, target_chunks = [], []
    source_chunk_ids, target_chunk_ids = [], []
    for (ss, se, ts, te) in spans:
        source_chunks.append(src_sep.join(src_words[ss:se]))
        target_chunks.append(tgt_sep.join(tgt_words[ts:te]))
        src_ids = [tid for span in src_word_spans[ss:se] for tid in span]
        tgt_ids = [tid for span in tgt_word_spans[ts:te] for tid in span]
        source_chunk_ids.append(src_ids)
        target_chunk_ids.append(tgt_ids)

    return {
        "index": row["index"],
        "source": src,
        "target": tgt,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "latency": latency,
        "source_chunks": source_chunks,
        "target_chunks": target_chunks,
        "source_chunk_ids": source_chunk_ids,
        "target_chunk_ids": target_chunk_ids,
        "_annotator_meta": {
            "criterion": "wait-k",
            "k": k,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="Experiment config YAML (must have annotate.criterion=wait-k "
                         "and annotate.wait_k_per_latency).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output SFT dataset JSON path.")
    ap.add_argument("--max_src_tokens", type=int, default=80,
                    help="Skip rows whose source tokenizes to > this many BPE ids. "
                         "Default 80 matches OT annotator to keep the training "
                         "distribution comparable across arms.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["annotate"]["criterion"] != "wait-k":
        sys.exit(f"config criterion is {cfg['annotate']['criterion']!r}, expected 'wait-k'")

    wait_k_map = cfg["annotate"].get("wait_k_per_latency")
    if not wait_k_map:
        sys.exit("config missing annotate.wait_k_per_latency (mapping latency→k)")

    latency_bins = cfg["annotate"]["latency_bins"]
    for lat in latency_bins:
        if lat not in wait_k_map:
            sys.exit(f"latency {lat!r} in latency_bins but not in wait_k_per_latency")
        if lat not in LATENCY_NL:
            sys.exit(f"latency {lat!r} not in supported NL ladder {LATENCY_NL}")

    tag = cfg["tag"]
    per_dir_root = REPO_ROOT / "results" / "sft_dataset" / tag / "per_direction"
    if not per_dir_root.exists():
        sys.exit(f"missing per-direction pool dir: {per_dir_root} "
                 f"(run stage 1: bin/run configs/{args.config.name} --stage 1)")

    from transformers import AutoTokenizer
    tok_path = cfg["backbone"]["tokenizer_dir"]
    print(f"Loading tokenizer from {tok_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    directions = list(cfg["source_pool"]["directions"].keys())
    print(f"Directions: {directions}")
    print(f"Wait-k per latency: {wait_k_map}")
    print(f"Max source tokens: {args.max_src_tokens}")

    all_rows: List[Dict] = []
    stats = {"total_input": 0, "kept_per_row": 0, "skipped_max_src": 0,
             "per_direction": {}, "per_latency": {lat: 0 for lat in latency_bins}}
    t0 = time.time()

    for pair in directions:
        pair_file = per_dir_root / f"{pair}.json"
        if not pair_file.exists():
            print(f"  [WARN] {pair}: missing {pair_file}, skipping", flush=True)
            continue
        with open(pair_file) as f:
            rows = json.load(f)
        n_input = len(rows)
        n_kept = 0
        n_skipped = 0
        for row in rows:
            base_ok = None
            for latency in latency_bins:
                k = int(wait_k_map[latency])
                built = build_row(row, tokenizer, k, latency, args.max_src_tokens)
                if built is None:
                    if base_ok is None:
                        base_ok = False
                    continue
                base_ok = True
                all_rows.append(built)
                stats["per_latency"][latency] += 1
            if base_ok:
                n_kept += 1
            else:
                n_skipped += 1
        stats["per_direction"][pair] = {"input": n_input, "kept": n_kept,
                                         "skipped": n_skipped}
        stats["total_input"] += n_input
        stats["kept_per_row"] += n_kept
        stats["skipped_max_src"] += n_skipped
        print(f"  {pair}: {n_input} input rows → {n_kept} kept × {len(latency_bins)} latencies"
              f" = {n_kept * len(latency_bins)} training rows; skipped {n_skipped}",
              flush=True)

    dt = time.time() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_rows, f, ensure_ascii=False)

    # Small companion summary next to the dataset.
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "tag": tag,
        "config": str(args.config),
        "output": str(args.output),
        "criterion": "wait-k",
        "wait_k_per_latency": wait_k_map,
        "latency_bins": latency_bins,
        "max_src_tokens": args.max_src_tokens,
        "n_output_rows": len(all_rows),
        "elapsed_sec": round(dt, 1),
        "stats": stats,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_rows):,} training rows to {args.output} in {dt:.1f}s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
