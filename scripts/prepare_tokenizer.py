"""Extend a backbone's tokenizer with EAST-format {EOR, EOW} special tokens.

v6 pivot: chat template + natural-language instruction replaces the vocab-token
latency indicators. Only EAST specials (end-of-read, end-of-write) need to be
in the vocab; latency and direction are natural-language in the user turn.

Usage:
    bin/prepare_tokenizer \\
        --backbone google/gemma-4-E2B-it \\
        --output   results/train/<tag>/tokenizer

Or with local paths:
    bin/prepare_tokenizer \\
        --backbone /path/to/gemma-4-E2B-it \\
        --output   /where/to/save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import MODEL_BASE

from transformers import AutoTokenizer

# v6 special tokens (only EOR/EOW — latency is NL now)
NEW_SPECIALS = ["<|end-of-read|>", "<|end-of-write|>"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backbone", required=True,
                    help="HF id or absolute path to the base tokenizer.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Directory to save the extended tokenizer to.")
    args = ap.parse_args()

    print(f"Loading base tokenizer from {args.backbone} ...")
    tok = AutoTokenizer.from_pretrained(args.backbone)
    print(f"  base vocab size: {tok.vocab_size}")
    print(f"  chat_template present: {tok.chat_template is not None}")

    added = tok.add_special_tokens({"additional_special_tokens": NEW_SPECIALS})
    print(f"Added {added} special tokens.")

    for t in NEW_SPECIALS:
        ids = tok(t, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"{t!r} splits into {len(ids)} ids: {ids}"
        print(f"  {t} → id {ids[0]}")

    # Round-trip test — chat template with EOR/EOW in assistant turn
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Translate the following text from English into German with low latency."},
        {"role": "assistant", "content": "Anyone with information<|end-of-read|> Jeder<|end-of-write|>"},
    ]
    full_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    print(f"\nSample chat-template rendered:")
    print(f"  {full_str!r}")
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    print(f"  n_ids = {len(ids)}")
    decoded = tok.decode(ids)
    print(f"  round-trip decode: {decoded!r}")

    args.output.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(args.output))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
