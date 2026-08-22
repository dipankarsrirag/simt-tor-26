"""
EAST-format interleaving (Fu et al. 2025, §3.2 and App. A / Fig. 18).

Shared between the annotator and the SFT dataset builder — one place
defines the special tokens and the interleave rule.

An EAST training example is:

    <latency> src_1 ... src_{i1} <|end-of-read|>
              tgt_1 ... tgt_{j1} <|end-of-write|>
              src_{i1+1} ... src_{i2} <|end-of-read|>
              tgt_{j1+1} ... tgt_{j2} <|end-of-write|>
              ...

where <latency> is one of {<|low-latency|>, <|medium-latency|>, <|high-latency|>}.
The final chunk's <|end-of-write|> ends the sequence.

The condition-A ("GPT-4 chunks") baseline gets these from the shipped
`source_chunks`/`target_chunks` fields. The condition-B ("ours") builder
constructs the same shape from the annotator's commit points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

END_OF_READ = "<|end-of-read|>"
END_OF_WRITE = "<|end-of-write|>"

# v1-v5 legacy: latency tokens injected into the vocabulary. Retained for
# backward compat with v1-v5 checkpoints only; v6+ replaces these with
# natural-language latency description in the chat prompt (matches EAST).
LATENCY_TOKENS = {
    "low": "<|low-latency|>",
    "medium": "<|medium-latency|>",
    "high": "<|high-latency|>",
}

SPECIAL_TOKENS = [END_OF_READ, END_OF_WRITE, *LATENCY_TOKENS.values()]

# v6+: only EOR/EOW are added to the vocabulary. Latency + direction are
# in the natural-language user turn of the chat template.
SPECIAL_TOKENS_V6 = [END_OF_READ, END_OF_WRITE]

# 5-point latency ladder in natural language. Training data has only the
# 3 base labels (low/medium/high); low-medium + medium-high are inference-
# only prompts (EAST §3.3 "interpolation effect").
LATENCY_NL = ["low", "low-medium", "medium", "medium-high", "high"]

# Language-code → English name (for the direction phrase in the chat prompt).
LANG_CODE_TO_NAME = {
    "en": "English", "de": "German", "ar": "Arabic",
    "ru": "Russian", "zh": "Chinese", "vi": "Vietnamese",
    "cs": "Czech", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "fr": "French",
    # accept full names too — pass-through
    "English": "English", "German": "German", "Arabic": "Arabic",
    "Russian": "Russian", "Chinese": "Chinese", "Vietnamese": "Vietnamese",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def lang_name(code: str) -> str:
    """Resolve a language code or name to its English name."""
    return LANG_CODE_TO_NAME.get(code, code)


def build_user_instruction(src_lang: str, tgt_lang: str, latency: str) -> str:
    """v6 chat prompt: natural-language translation instruction.

    Matches EAST's actual prompt style (Fu et al. 2025, Fig. 18):
      'Translate the following text from {src_lang_name} into
       {tgt_lang_name} with {latency} latency.'

    where latency is a 5-point NL ladder: low, low-medium, medium,
    medium-high, high. Training uses 3 base values; inference gets the
    interpolated points free.
    """
    if latency not in LATENCY_NL:
        raise ValueError(f"unknown latency {latency!r}; expected one of {LATENCY_NL}")
    return (f"Translate the following text from {lang_name(src_lang)} into "
            f"{lang_name(tgt_lang)} with {latency} latency.")


def build_assistant_body(source_chunks: List[str], target_chunks: List[str],
                          src_lang: str = "en", tgt_lang: str = "en") -> str:
    """v6 assistant-turn body: interleaved EAST-format chunks.

      src_1 <|end-of-read|> tgt_1 <|end-of-write|>
      src_2 <|end-of-read|> tgt_2 <|end-of-write|>
      ...

    Leading-space-per-chunk convention preserved from v4/v5 (fixes phantom
    `▁` bug). For CJK sides (zh/ja/ko/th/km), no leading space between chunks
    since those scripts have no whitespace word boundaries.
    """
    from src.annotator.annotate import _is_cjk_lang  # local to avoid cycles
    src_sep = "" if _is_cjk_lang(src_lang) else " "
    tgt_sep = "" if _is_cjk_lang(tgt_lang) else " "
    parts = []
    for i, (src_c, tgt_c) in enumerate(zip(source_chunks, target_chunks)):
        parts.append((src_sep if i == 0 else src_sep) + src_c.strip())
        parts.append(END_OF_READ)
        parts.append(tgt_sep + tgt_c.strip())
        parts.append(END_OF_WRITE)
    return "".join(parts)


def build_chat_prompt_v6(row: EastRow, tokenizer, add_generation_prompt: bool = False,
                          system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """v6 renderer: Gemma-4-it chat template + EAST-format assistant body.

    Produces the full chat string ready for tokenization. If
    `add_generation_prompt=True`, the assistant turn is left open (for
    inference); else, the assistant turn is closed (for training).
    """
    user_instr = build_user_instruction(row.src_lang, row.tgt_lang, row.latency)
    assistant_body = build_assistant_body(
        row.source_chunks, row.target_chunks, row.src_lang, row.tgt_lang
    )
    messages = [
        {"role": "user", "content": user_instr},
        {"role": "assistant", "content": assistant_body},
    ]
    # Gemma-4-it doesn't always accept a system role in the template — merge
    # system into user turn if the template rejects it. Gemma-4-it does accept
    # system in the version we probed (returned system/user/model sequence).
    if system_prompt:
        try:
            _ = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}] + messages,
                tokenize=False, add_generation_prompt=False,
            )
            messages = [{"role": "system", "content": system_prompt}] + messages
        except Exception:
            # Fold into user turn
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


@dataclass
class EastRow:
    source: str
    target: str
    src_lang: str
    tgt_lang: str
    latency: str  # "low" | "medium" | "high"
    source_chunks: List[str]
    target_chunks: List[str]


def interleave(row: EastRow, chunk_sep: str = " ", fixed_tokenization: bool = False) -> str:
    """Produce the EAST training string.

    Legacy (default): `chunk_sep=" "` joins all parts with a space. This
    produces `<|low-latency|> src <|end-of-read|> tgt <|end-of-write|>`
    but re-tokenization introduces a standalone `▁` before every EAST
    special (since SentencePiece treats special tokens as boundaries and
    the surrounding space becomes its own token). The model then learns
    `content → ▁ → EOR` and at inference (streaming feeds words directly
    without the standalone ▁) `p(EOR)` never fires.

    `fixed_tokenization=True` (2026-08-19): prepend a space to each chunk
    (so its first BPE gets the SentencePiece `▁` prefix naturally) and
    join with EMPTY string. The training string becomes
    `<|low-latency|> src<|end-of-read|> tgt<|end-of-write|> ...` where
    the leading space on each chunk IS the natural word-boundary marker
    and specials sit DIRECTLY after the last BPE of the previous chunk —
    no standalone ▁ separator. Streaming-inference feeding a source word
    then queries argmax at the same position where training taught EOR.
    """
    if len(row.source_chunks) != len(row.target_chunks):
        raise ValueError(
            f"chunk-count mismatch: {len(row.source_chunks)} src vs "
            f"{len(row.target_chunks)} tgt (EAST discards these — App. C)"
        )
    if row.latency not in LATENCY_TOKENS:
        raise ValueError(f"unknown latency {row.latency!r}; expected one of {list(LATENCY_TOKENS)}")

    if fixed_tokenization:
        # Leading-space-per-chunk + empty join. Each chunk's first BPE gets
        # its ▁ marker from the leading space; EAST specials attach directly
        # to the last BPE of the previous chunk (no phantom ▁ separator).
        # .strip() defensive: prevents double-space if a chunk has leading
        # whitespace from upstream decode artifacts (2026-08-19).
        parts = [LATENCY_TOKENS[row.latency]]
        for src_c, tgt_c in zip(row.source_chunks, row.target_chunks):
            parts.append(" " + src_c.strip())
            parts.append(END_OF_READ)
            parts.append(" " + tgt_c.strip())
            parts.append(END_OF_WRITE)
        return "".join(parts)

    # Legacy path — kept for reproducing v1/v2/v3 datasets exactly.
    parts = [LATENCY_TOKENS[row.latency]]
    for src_c, tgt_c in zip(row.source_chunks, row.target_chunks):
        parts.extend([src_c, END_OF_READ, tgt_c, END_OF_WRITE])
    return chunk_sep.join(parts)


def parse_row(d: dict) -> EastRow:
    """Lift a dict (as shipped in SiMT-De-En-660K.json) into an EastRow."""
    return EastRow(
        source=d["source"],
        target=d["target"],
        src_lang=d["src_lang"],
        tgt_lang=d["tgt_lang"],
        latency=d["latency"],
        source_chunks=list(d["source_chunks"]),
        target_chunks=list(d["target_chunks"]),
    )
