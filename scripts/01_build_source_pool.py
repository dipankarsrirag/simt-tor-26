"""
Build the v6b-curated multi-direction source pool with HUMAN-translated targets
only. Drops the GPT-4-translated Multi-90K pool for de/ru pairs and replaces
with europarl / news-commentary / TED2020 (all human translations).

Rationale: our chunker teacher-freeness (OT vs GPT-4) is already established
by v2bal_v3. This build additionally removes the target-translation teacher —
so the final model has NO GPT-4 dependency at any stage.

Sources per pair:
  de-en, en-de : europarl-v8 + news-commentary-v16 + ted2020-v1 (already on disk)
  ru-en, en-ru : OPUS TED2020-v1 + news-commentary-v16 (downloaded separately)
  ar-en, en-ar : TED2020 (unchanged from v6b — was already human-translated)
  vi-en, en-vi : TED2020 (unchanged from v6b — was already human-translated)

Output schema matches phase2_build_multilingual_source_pool.py — same row
shape so the annotator template runs unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DATA_ROOT

# Corpus roots. Overridable via CLI (--deen_root, --ruen_root, etc.) or via
# SIMT_CORPUS_ROOT env var. Default layout assumes DATA_ROOT/parallel_clean/
# and DATA_ROOT/raw/ (matches the shared sibling repo simul-mt/).
import os
CORPUS_ROOT = Path(os.environ.get("SIMT_CORPUS_ROOT", DATA_ROOT))
DEEN_ROOT = CORPUS_ROOT / "parallel_clean" / "de-en"
RUEN_ROOT = CORPUS_ROOT / "parallel_clean" / "ru-en"
AREN_ROOT = CORPUS_ROOT / "parallel_clean" / "ar-en"
VIEN_TED = {
    "src": CORPUS_ROOT / "raw" / "ted2020-en-vi" / "TED2020.en-vi.vi",
    "tgt": CORPUS_ROOT / "raw" / "ted2020-en-vi" / "TED2020.en-vi.en",
}

# Corpus preference order per direction. First-hit wins per row up to quota.
# All human translations, no GPT-4 anywhere.
DEEN_CORPORA = ["news-commentary", "ted2020", "europarl"]  # smallest first for domain diversity
RUEN_CORPORA = ["news-commentary", "ted2020"]              # opensubtitles avoided (informal)

# 8 directions to sample (10 in v5, but zh dropped in v6b + not needed for curated)
DIRECTIONS = [
    ("de", "en"), ("en", "de"),
    ("ar", "en"), ("en", "ar"),
    ("ru", "en"), ("en", "ru"),
    ("vi", "en"), ("en", "vi"),
]

# Language full-name map (for the row schema)
LANG_NAME = {
    "de": "German", "en": "English", "ru": "Russian", "ar": "Arabic",
    "vi": "Vietnamese",
}


def read_lines(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f]


def load_pair(root: Path, base_name: str, src_lang: str, tgt_lang: str) -> List[Dict]:
    src_path = root / f"{base_name}.{src_lang}"
    tgt_path = root / f"{base_name}.{tgt_lang}"
    if not src_path.exists() or not tgt_path.exists():
        return []
    srcs = read_lines(src_path)
    tgts = read_lines(tgt_path)
    n = min(len(srcs), len(tgts))
    rows = []
    for i in range(n):
        if srcs[i] and tgts[i]:
            rows.append({"source": srcs[i], "target": tgts[i],
                         "src_lang": src_lang, "tgt_lang": tgt_lang,
                         "_corpus": base_name})
    return rows


def load_ted_flat(files: Dict[str, Path], src_lang: str, tgt_lang: str) -> List[Dict]:
    if not files["src"].exists() or not files["tgt"].exists():
        return []
    srcs = read_lines(files["src"])
    tgts = read_lines(files["tgt"])
    n = min(len(srcs), len(tgts))
    rows = []
    for i in range(n):
        if srcs[i] and tgts[i]:
            rows.append({"source": srcs[i], "target": tgts[i],
                         "src_lang": src_lang, "tgt_lang": tgt_lang,
                         "_corpus": "ted2020"})
    return rows


def build_direction(src: str, tgt: str, n_target: int, seed: int) -> List[Dict]:
    """Sample n_target human-translated (src, tgt) rows for this direction."""
    key = (src, tgt)
    all_rows: List[Dict] = []

    if key == ("de", "en") or key == ("en", "de"):
        for base in DEEN_CORPORA:
            pair = load_pair(DEEN_ROOT, base, "de", "en")
            if not pair: continue
            # Reverse if needed
            if src == "en":
                pair = [{"source": r["target"], "target": r["source"],
                         "src_lang": "en", "tgt_lang": "de",
                         "_corpus": r["_corpus"]} for r in pair]
            all_rows.extend(pair)

    elif key == ("ru", "en") or key == ("en", "ru"):
        for base in RUEN_CORPORA:
            pair = load_pair(RUEN_ROOT, base, "en", "ru")
            if not pair:
                pair = load_pair(RUEN_ROOT, base, "ru", "en")  # try alternate order
                if not pair: continue
                if src == "en":
                    pair = [{"source": r["target"], "target": r["source"],
                             "src_lang": "en", "tgt_lang": "ru",
                             "_corpus": r["_corpus"]} for r in pair]
            else:
                # loaded en/ru; swap if we need ru->en
                if src == "ru":
                    pair = [{"source": r["target"], "target": r["source"],
                             "src_lang": "ru", "tgt_lang": "en",
                             "_corpus": r["_corpus"]} for r in pair]
            all_rows.extend(pair)

    elif key == ("ar", "en") or key == ("en", "ar"):
        # ar-en TED2020 already on disk (was used in v6b)
        pair = load_pair(AREN_ROOT, "ted2020", "ar", "en")
        if src == "en":
            pair = [{"source": r["target"], "target": r["source"],
                     "src_lang": "en", "tgt_lang": "ar",
                     "_corpus": r["_corpus"]} for r in pair]
        all_rows.extend(pair)

    elif key == ("vi", "en") or key == ("en", "vi"):
        # vi-en TED2020 flat files
        pair = load_ted_flat(VIEN_TED, "vi", "en")
        if src == "en":
            pair = [{"source": r["target"], "target": r["source"],
                     "src_lang": "en", "tgt_lang": "vi",
                     "_corpus": r["_corpus"]} for r in pair]
        all_rows.extend(pair)

    if not all_rows:
        return []

    # Deduplicate on source string
    seen = set(); dedup = []
    for r in all_rows:
        if r["source"] in seen: continue
        seen.add(r["source"]); dedup.append(r)
    print(f"  {src}-{tgt}: pool={len(all_rows):,} unique_src={len(dedup):,}", flush=True)

    # Filter by rough length band (5..80 whitespace tokens on source)
    dedup = [r for r in dedup
             if 5 <= len(r["source"].split()) <= 80
             and 5 <= len(r["target"].split()) <= 100]
    print(f"    length-filtered: {len(dedup):,}")

    # Shuffle and sample
    rng = random.Random(seed + hash((src, tgt)) % 10000)
    rng.shuffle(dedup)
    picked = dedup[:n_target]
    print(f"    sampled: {len(picked):,}/{n_target}")

    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_dir", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results" / "phase2" / "multilingual_source_pool_htgt_per_direction")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated dirs to build (e.g. 'de-en,en-de'). "
                         "Default: all 8. Useful for partial runs (e.g. defer ru until download).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(",")) if args.only else None

    global_index = 0
    for src, tgt in DIRECTIONS:
        pair_id = f"{src}-{tgt}"
        if want is not None and pair_id not in want:
            continue
        print(f"\n=== {pair_id} ===")
        rows = build_direction(src, tgt, args.n_per_dir, args.seed)
        # Emit in the schema expected by phase1_tau_sweep.py (indexed rows).
        out_rows = []
        for r in rows:
            out_rows.append({
                "index": global_index,
                "source": r["source"],
                "target": r["target"],
                "src_lang": LANG_NAME.get(src, src),
                "tgt_lang": LANG_NAME.get(tgt, tgt),
                "latency": "medium",  # placeholder
                "direction": pair_id,
                "source_chunks": [],
                "target_chunks": [],
                "_corpus": r["_corpus"],
            })
            global_index += 1
        out_path = args.out_dir / f"{pair_id}.json"
        with open(out_path, "w") as f:
            json.dump(out_rows, f, ensure_ascii=False)
        print(f"  wrote {out_path} ({len(out_rows):,} rows)")

    # Summary of corpora used
    print("\n=== Summary ===")
    for src, tgt in DIRECTIONS:
        p = args.out_dir / f"{src}-{tgt}.json"
        if not p.exists(): continue
        rows = json.loads(p.read_text())
        from collections import Counter
        c = Counter(r["_corpus"] for r in rows)
        print(f"  {src}-{tgt}: {len(rows):,} rows  {dict(c)}")


if __name__ == "__main__":
    main()
