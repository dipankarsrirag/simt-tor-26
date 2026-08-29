"""
Prepare the extended tokenizer for Phase 2.

Adds the five EAST special tokens to Gemma-4-E2B's tokenizer and saves the
result to a versioned path. Downstream (SFT + inference) MUST load from
this path — not from MODEL_BASE/gemma-4-E2B/ — or subtle tokenization
drift will break every metric (advisor blocker).

Special tokens (see src/annotator/east_format.py):
  <|end-of-read|>, <|end-of-write|>,
  <|low-latency|>, <|medium-latency|>, <|high-latency|>

Also verifies each token round-trips exactly (encode(tok) -> [id]; decode([id]) -> tok).
Prints new vocabulary size so the SFT wrapper knows how much to resize embeddings.

Usage:
    python scripts/phase2_prepare_tokenizer.py
    # then use results/phase2/tokenizer-extended/ everywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.east_format import SPECIAL_TOKENS
from src.constants import PRIMARY_BACKBONE, REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tokenizer", type=str, default=str(PRIMARY_BACKBONE))
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "results" / "phase2" / "tokenizer-extended")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    print(f"Loading base tokenizer from {args.base_tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    original_vocab = len(tok)
    print(f"  base vocab size: {original_vocab:,}")

    print(f"Adding EAST special tokens: {SPECIAL_TOKENS}")
    n_added = tok.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS)}
    )
    new_vocab = len(tok)
    print(f"  added {n_added} tokens; new vocab size: {new_vocab:,}")

    # Verify each special token round-trips.
    for t in SPECIAL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        assert len(ids) == 1, (
            f"{t!r} did not encode to a single token: got {ids}. "
            "add_special_tokens may have failed."
        )
        rt = tok.decode(ids, skip_special_tokens=False)
        assert rt == t, f"round-trip failed for {t!r}: got {rt!r}"
        print(f"  {t!r} -> id {ids[0]}")

    print(f"\nSaving extended tokenizer to {args.output_dir} ...")
    tok.save_pretrained(args.output_dir)

    # Manifest — the SFT wrapper reads this to know how many embeddings to add.
    manifest = {
        "base_tokenizer": args.base_tokenizer,
        "base_vocab_size": original_vocab,
        "added_tokens": list(SPECIAL_TOKENS),
        "new_vocab_size": new_vocab,
        "n_added": n_added,
        "special_token_ids": {
            t: tok.encode(t, add_special_tokens=False)[0] for t in SPECIAL_TOKENS
        },
    }
    (args.output_dir / "east_special_tokens.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(f"  wrote manifest: {args.output_dir}/east_special_tokens.json")

    # Reload sanity check.
    print(f"\nReloading from saved path to confirm persistence ...")
    tok2 = AutoTokenizer.from_pretrained(args.output_dir)
    assert len(tok2) == new_vocab
    for t in SPECIAL_TOKENS:
        ids = tok2.encode(t, add_special_tokens=False)
        assert len(ids) == 1, f"reload broke {t!r}: {ids}"
    print("  reload OK")

    print(f"\nDONE. new_vocab_size = {new_vocab:,} "
          f"(base {original_vocab:,} + {n_added} added).")
    print(f"Downstream: pass tokenizer_path={args.output_dir} to SFT and inference.")


if __name__ == "__main__":
    main()
