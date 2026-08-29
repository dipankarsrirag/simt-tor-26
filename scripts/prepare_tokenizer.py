"""Build the v6 extended tokenizer: gemma-4-E2B-it + {EOR, EOW}.

v6 pivot: chat template + natural-language instruction replaces the vocab-token
latency indicators. Only EAST specials (end-of-read, end-of-write) need to be
in the vocab; latency and direction are natural-language in the user turn.

Output: /g/data/ba39/dipankar/simt-tor-26/_archive/results/v6b_gemma_2b/tokenizer-extended-v6
"""
from pathlib import Path
from transformers import AutoTokenizer

SRC_TOK = "/g/data/po67/dipankar/models/gemma-4-E2B-it"
OUT_DIR = Path("/g/data/ba39/dipankar/simt-tor-26/_archive/results/v6b_gemma_2b/tokenizer-extended-v6")

# v6 special tokens (only EOR/EOW — latency is NL now)
NEW_SPECIALS = ["<|end-of-read|>", "<|end-of-write|>"]

print(f"Loading base tokenizer from {SRC_TOK} ...")
tok = AutoTokenizer.from_pretrained(SRC_TOK)
print(f"  base vocab size: {tok.vocab_size}")
print(f"  chat_template present: {tok.chat_template is not None}")

added = tok.add_special_tokens({"additional_special_tokens": NEW_SPECIALS})
print(f"Added {added} special tokens.")

# Verify each maps to exactly 1 id
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

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(str(OUT_DIR))
print(f"\nSaved to {OUT_DIR}")
