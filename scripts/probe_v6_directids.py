"""Probe direct-ids splice strategy for v6 training.

Strategy: use a unique placeholder in the chat-template render, split, tokenize
each side, concatenate: prefix_ids + assistant_ids_from_chunks + suffix_ids.

Verify:
  1. Placeholder split is clean (both halves tokenize unambiguously).
  2. Prefix_ids ends with '<|turn>model\n' tokens; suffix starts with '<turn|>'.
  3. When we splice in a small synthetic assistant body (chunk ids), the
     resulting sequence decodes to something coherent.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from transformers import AutoTokenizer
from src.annotator.east_format import (
    END_OF_READ, END_OF_WRITE, build_user_instruction, DEFAULT_SYSTEM_PROMPT
)

TOK_DIR = "/g/data/ba39/dipankar/simt-tor-26/results/phase2/tokenizer-extended-v6"
PLACEHOLDER = "PLACEHOLDER_ASSISTANT_BODY"


def render_open_close(tok, src_lang, tgt_lang, latency):
    user_instr = build_user_instruction(src_lang, tgt_lang, latency)
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_instr},
        {"role": "assistant", "content": PLACEHOLDER},
    ]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        messages = [
            {"role": "user", "content": DEFAULT_SYSTEM_PROMPT + "\n\n" + user_instr},
            {"role": "assistant", "content": PLACEHOLDER},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if PLACEHOLDER not in text:
        raise RuntimeError("placeholder not present in rendered chat text")
    open_str, close_str = text.split(PLACEHOLDER)
    return open_str, close_str, text


def main():
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    eor_id = tok(END_OF_READ, add_special_tokens=False).input_ids[0]
    eow_id = tok(END_OF_WRITE, add_special_tokens=False).input_ids[0]
    print(f"EOR id = {eor_id}, EOW id = {eow_id}")

    for src_lang, tgt_lang in [("ar", "en"), ("vi", "en"), ("de", "en"), ("en", "de")]:
        print(f"\n=== {src_lang}-{tgt_lang} ===")
        open_str, close_str, full = render_open_close(tok, src_lang, tgt_lang, "medium")
        print(f"  open_str  ends with: {open_str[-40:]!r}")
        print(f"  close_str starts with: {close_str[:40]!r}")

        prefix_ids = tok(open_str, add_special_tokens=False)["input_ids"]
        suffix_ids = tok(close_str, add_special_tokens=False)["input_ids"]
        print(f"  prefix_ids last 6 tokens: {[tok.convert_ids_to_tokens(t) for t in prefix_ids[-6:]]}")
        print(f"  suffix_ids first 6 tokens: {[tok.convert_ids_to_tokens(t) for t in suffix_ids[:6]]}")

        # Synthetic single-chunk assistant body: source_ids + [eor] + target_ids + [eow]
        # Take source from a real string, tokenize once (annotator style).
        src = "Wenden Sie sich bitte an" if src_lang == "de" else \
              "Please contact your local representative" if src_lang == "en" else \
              "من فضلك اتصل" if src_lang == "ar" else \
              "Vui lòng liên hệ" if src_lang == "vi" else "test"
        tgt = "Please contact your local representative" if src_lang != "en" else \
              "Bitte wenden Sie sich"

        src_ids = tok(src, add_special_tokens=False)["input_ids"]
        tgt_ids = tok(tgt, add_special_tokens=False)["input_ids"]
        # Split source in half for a 2-chunk demo
        mid = len(src_ids) // 2
        src_chunks = [src_ids[:mid], src_ids[mid:]]
        # Split target in half also
        midt = len(tgt_ids) // 2
        tgt_chunks = [tgt_ids[:midt], tgt_ids[midt:]]

        assistant_ids = []
        for s, t in zip(src_chunks, tgt_chunks):
            assistant_ids.extend(s)
            assistant_ids.append(eor_id)
            assistant_ids.extend(t)
            assistant_ids.append(eow_id)

        full_ids = prefix_ids + assistant_ids + suffix_ids
        # Decode to make sure it looks like a coherent chat
        decoded = tok.decode(full_ids, skip_special_tokens=False)
        # Just print the ASSISTANT part (from <|turn>model onwards)
        marker = "<|turn>model\n"
        idx = decoded.find(marker)
        assistant_part = decoded[idx:] if idx >= 0 else "(no marker)"
        print(f"  spliced assistant (first 200): {assistant_part[:200]!r}")

        # Also sanity check: verify prefix does NOT contain EOR/EOW (else masking is broken)
        assert eor_id not in prefix_ids, "EOR leaked into prefix!"
        assert eow_id not in prefix_ids, "EOW leaked into prefix!"
        # And suffix does not contain them
        assert eor_id not in suffix_ids
        assert eow_id not in suffix_ids
        print("  OK: no EOR/EOW leak into prefix/suffix")


if __name__ == "__main__":
    main()
