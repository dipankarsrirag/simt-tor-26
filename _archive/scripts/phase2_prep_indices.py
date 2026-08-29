"""
Pre-compute deterministic latency-balanced indices for Phase-2 SFT + cond-B
annotation. Same protocol as src/train/sft.py's pick_latency_balanced but
saved once so both arms + annotation use the exact same rows.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.constants import DATA_ROOT, PRIMARY_BACKBONE, REPO_ROOT

CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_src_tokens", type=int, default=80)
    ap.add_argument("--tokenizer_path", default=str(PRIMARY_BACKBONE))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading corpus from {CORPUS}", flush=True)
    with open(CORPUS) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} rows", flush=True)

    rng = random.Random(args.seed)
    by_lat = {}
    for r in rows:
        by_lat.setdefault(r["latency"], []).append(r)
    per = args.n // 3
    picked = []
    for lat in ["low", "medium", "high"]:
        pool = by_lat.get(lat, [])
        rng.shuffle(pool)
        picked.extend(pool[:per])
    remainder = args.n - len(picked)
    if remainder > 0:
        picked.extend(by_lat["medium"][per: per + remainder])
    print(f"picked={len(picked)} (n_requested={args.n})", flush=True)

    # Apply length + chunk-count filters (same as sft.py).
    kept = []
    total_toks = 0
    for r in picked:
        n_src = len(tok(r["source"], add_special_tokens=False)["input_ids"])
        if n_src > args.max_src_tokens:
            continue
        if len(r.get("source_chunks", [])) != len(r.get("target_chunks", [])):
            continue
        if not r.get("source_chunks"):
            continue
        kept.append(r)
        total_toks += n_src
    print(f"kept={len(kept)} (dropped={len(picked)-len(kept)} for length+chunk-count filters)", flush=True)

    indices = sorted(r["index"] for r in kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "seed": args.seed,
        "n_requested": args.n,
        "n_kept": len(kept),
        "filter": {"max_src_tokens": args.max_src_tokens, "chunk_count_match": True},
        "source": "pick_latency_balanced from src/train/sft.py",
        "indices": indices,
        "by_latency": {
            lat: sorted(r["index"] for r in kept if r["latency"] == lat)
            for lat in ["low", "medium", "high"]
        },
    }, indent=2))
    print(f"Wrote {args.output}", flush=True)
    from collections import Counter
    lat_counts = Counter(r["latency"] for r in kept)
    print(f"Latency counts: {dict(lat_counts)}", flush=True)


if __name__ == "__main__":
    main()
