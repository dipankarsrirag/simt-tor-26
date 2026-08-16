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

from dataclasses import dataclass
from typing import List

END_OF_READ = "<|end-of-read|>"
END_OF_WRITE = "<|end-of-write|>"

LATENCY_TOKENS = {
    "low": "<|low-latency|>",
    "medium": "<|medium-latency|>",
    "high": "<|high-latency|>",
}

SPECIAL_TOKENS = [END_OF_READ, END_OF_WRITE, *LATENCY_TOKENS.values()]


@dataclass
class EastRow:
    source: str
    target: str
    src_lang: str
    tgt_lang: str
    latency: str  # "low" | "medium" | "high"
    source_chunks: List[str]
    target_chunks: List[str]


def interleave(row: EastRow, chunk_sep: str = " ") -> str:
    """Produce the EAST training string. chunk_sep is what glues tokens
    within a chunk when the chunks are already whitespace-tokenised in
    the corpus (they are)."""
    if len(row.source_chunks) != len(row.target_chunks):
        raise ValueError(
            f"chunk-count mismatch: {len(row.source_chunks)} src vs "
            f"{len(row.target_chunks)} tgt (EAST discards these — App. C)"
        )
    if row.latency not in LATENCY_TOKENS:
        raise ValueError(f"unknown latency {row.latency!r}; expected one of {list(LATENCY_TOKENS)}")

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
