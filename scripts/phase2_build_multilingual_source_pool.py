"""Build unified multi-direction source pool for v5 OT annotation.

Samples ~10K rows per direction (10 directions total: {de,ar,ru,zh,vi}↔en)
and emits a single JSON in the unified schema consumed by the OT annotator.

Sources:
  de↔en, ru↔en, zh↔en  : SiMT-Multi-90K (has GPT-4 chunks already but we'll
                          re-annotate with OT; strip incoming chunks to avoid
                          confusion, keep only src/tgt/lang/latency-placeholder)
  ar↔en                : /g/data/ba39/dipankar/simul-mt/data/parallel_clean/
                          ar-en/ted2020.{ar,en} (line-aligned raw pairs)
  vi↔en                : /g/data/ba39/dipankar/simul-mt/data/raw/
                          ted2020-en-vi/TED2020.en-vi.{en,vi}

Output schema (per row):
  {
    "index":        int,          # unique across all directions
    "source":       str,          # cleaned (strip)
    "target":       str,          # cleaned (strip)
    "src_lang":     str,          # short code: de, en, ar, ru, zh, vi
    "tgt_lang":     str,
    "latency":      str,          # "medium" placeholder — reassigned post-annot
    "direction":    str,          # "de-en", "en-de", ...
    "source_chunks": [],          # empty — annotator fills these
    "target_chunks": [],
  }

Length filter: source ≤ MAX_SRC_TOKENS after Gemma-4 tokenization (matches
existing pipeline). Rows exceeding are dropped and NOT counted toward the
per-direction quota — we resample from the remaining pool until N is met
or the source pool is exhausted (zh may be short since Multi-90K has only
~8K per direction).
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Source data locations
MULTI_90K = Path("/g/data/po67/dipankar/data/simt-tor-26/SiMT-Multi-90K/SiMT-Multi-90K.json")
AR_TED = {
    "src": Path("/g/data/ba39/dipankar/simul-mt/data/parallel_clean/ar-en/ted2020.ar"),
    "tgt": Path("/g/data/ba39/dipankar/simul-mt/data/parallel_clean/ar-en/ted2020.en"),
}
VI_TED = {
    "src": Path("/g/data/ba39/dipankar/simul-mt/data/raw/ted2020-en-vi/TED2020.en-vi.vi"),
    "tgt": Path("/g/data/ba39/dipankar/simul-mt/data/raw/ted2020-en-vi/TED2020.en-vi.en"),
}

# Multi-90K language-name → project short-code map
LANG_CODE = {
    "English": "en", "German": "de", "Russian": "ru",
    "Chinese": "zh", "Czech": "cs",
}

# 10 directions to sample
DIRECTIONS = [
    ("de", "en"), ("en", "de"),
    ("ar", "en"), ("en", "ar"),
    ("ru", "en"), ("en", "ru"),
    ("zh", "en"), ("en", "zh"),
    ("vi", "en"), ("en", "vi"),
]


def load_multi90k_rows():
    """Load Multi-90K, produce dict: (src_code, tgt_code) → [rows]."""
    print(f"Loading Multi-90K from {MULTI_90K} ...", flush=True)
    with open(MULTI_90K) as f:
        rows = json.load(f)
    by_dir = defaultdict(list)
    for r in rows:
        s = LANG_CODE.get(r["src_lang"])
        t = LANG_CODE.get(r["tgt_lang"])
        if s is None or t is None:
            continue
        by_dir[(s, t)].append({
            "source": r["source"].strip(),
            "target": r["target"].strip(),
            "src_lang": s,
            "tgt_lang": t,
        })
    for k, v in sorted(by_dir.items()):
        print(f"  {k}: {len(v)} rows")
    return by_dir


def load_line_aligned(src_path: Path, tgt_path: Path, src_lang: str, tgt_lang: str, limit: int = None):
    """Load line-aligned raw parallel files → row dicts."""
    print(f"Loading {src_path.name} + {tgt_path.name} ...", flush=True)
    with open(src_path) as fs, open(tgt_path) as ft:
        src_lines = [l.strip() for l in fs]
        tgt_lines = [l.strip() for l in ft]
    assert len(src_lines) == len(tgt_lines), f"line-count mismatch: {len(src_lines)} vs {len(tgt_lines)}"
    rows = []
    for s, t in zip(src_lines, tgt_lines):
        if not s or not t:
            continue
        rows.append({
            "source": s, "target": t,
            "src_lang": src_lang, "tgt_lang": tgt_lang,
        })
        if limit and len(rows) >= limit:
            break
    print(f"  kept {len(rows)} pairs")
    return rows


def make_flipped(rows: List[Dict]) -> List[Dict]:
    """Flip src/tgt of every row (for building en→X from X→en raw data)."""
    return [{
        "source": r["target"], "target": r["source"],
        "src_lang": r["tgt_lang"], "tgt_lang": r["src_lang"],
    } for r in rows]


def sample_direction(src_lang: str, tgt_lang: str, pool: List[Dict],
                     n_target: int, rng: random.Random, tokenizer, max_src_tokens: int):
    """Sample up to n_target rows from pool, filtered by src length."""
    if not pool:
        print(f"  ⚠️  no pool for {src_lang}→{tgt_lang}")
        return []
    rng.shuffle(pool)
    kept = []
    for r in pool:
        n_src = len(tokenizer(r["source"], add_special_tokens=False)["input_ids"])
        if n_src == 0 or n_src > max_src_tokens:
            continue
        kept.append({**r, "latency": "medium",
                     "direction": f"{src_lang}-{tgt_lang}",
                     "source_chunks": [], "target_chunks": []})
        if len(kept) >= n_target:
            break
    print(f"  {src_lang}→{tgt_lang}: kept {len(kept)}/{n_target}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_direction", type=int, default=10000)
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer_dir", type=str,
                    default=str(REPO / "results/phase2/tokenizer-extended"))
    ap.add_argument("--output", type=str,
                    default=str(REPO / "results/phase2/multilingual_source_pool_v5.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    rng = random.Random(args.seed)

    # Load raw pools
    multi = load_multi90k_rows()
    ar_pool = load_line_aligned(AR_TED["src"], AR_TED["tgt"], "ar", "en", limit=args.n_per_direction * 5)
    vi_pool = load_line_aligned(VI_TED["src"], VI_TED["tgt"], "vi", "en", limit=args.n_per_direction * 5)

    # Build per-direction pool → sample
    pool_by_dir: Dict[tuple, List[Dict]] = {}
    for (s, t) in DIRECTIONS:
        if (s, t) == ("ar", "en"):
            pool_by_dir[(s, t)] = list(ar_pool)
        elif (s, t) == ("en", "ar"):
            pool_by_dir[(s, t)] = make_flipped(ar_pool)
        elif (s, t) == ("vi", "en"):
            pool_by_dir[(s, t)] = list(vi_pool)
        elif (s, t) == ("en", "vi"):
            pool_by_dir[(s, t)] = make_flipped(vi_pool)
        else:
            pool_by_dir[(s, t)] = multi.get((s, t), [])

    all_sampled = []
    global_idx = 0
    per_direction_counts = {}
    for (s, t) in DIRECTIONS:
        rows = sample_direction(s, t, pool_by_dir[(s, t)],
                                args.n_per_direction, rng, tok, args.max_src_tokens)
        for r in rows:
            r["index"] = global_idx
            global_idx += 1
        all_sampled.extend(rows)
        per_direction_counts[f"{s}-{t}"] = len(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_sampled, f, ensure_ascii=False)

    print(f"\nWrote {len(all_sampled)} rows to {out_path}")
    print(f"Per-direction:")
    for d, n in per_direction_counts.items():
        print(f"  {d}: {n}")

    # Also dump per-direction JSONs — the annotator can consume any of these
    # directly via --input_json.
    per_dir_dir = out_path.parent / "multilingual_source_pool_v5_per_direction"
    per_dir_dir.mkdir(exist_ok=True)
    by_dir: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_sampled:
        by_dir[r["direction"]].append(r)
    for d, rows in by_dir.items():
        p = per_dir_dir / f"{d}.json"
        with open(p, "w") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"  wrote per-dir file: {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
