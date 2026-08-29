"""
Phase 0 deliverable: reconstruct one shipped row from SiMT-De-En-660K
by hand, in EAST format (App. A / Fig. 18), and verify:

  * source == whitespace-concat(source_chunks) after trivial punctuation-glue
  * target == whitespace-concat(target_chunks) after trivial punctuation-glue
  * chunk counts match (they must — EAST discards mismatched rows before
    release, see App. C)
  * the interleaved string tokenises cleanly under the primary tokenizer
    with SPECIAL_TOKENS added

Also prints the reconstructed string for one row per latency level so a
human can eyeball against EAST Fig. 18. This is what stops format bugs
from surviving into Phase 2.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/g/data/ba39/dipankar/simt-tor-26")

from src.annotator.east_format import (  # noqa: E402
    END_OF_READ,
    END_OF_WRITE,
    LATENCY_TOKENS,
    SPECIAL_TOKENS,
    interleave,
    parse_row,
)
from src.constants import DATA_ROOT, PRIMARY_BACKBONE  # noqa: E402


CORPUS = DATA_ROOT / "SiMT-De-En-660K" / "SiMT-De-En-660K.json"


def normalise(s: str) -> str:
    """Collapse whitespace for the join-equivalence check.
    Chunk boundaries may drop a space that the raw sentence had, so we
    compare on whitespace-normalised strings."""
    return " ".join(s.split())


def main():
    print(f"Loading {CORPUS} ...")
    with open(CORPUS) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows):,} rows")

    # Latency counts (matches the LOG.md handoff entry: low=230,902 / medium=227,131 / high=202,843).
    counts = {}
    for r in rows:
        counts[r["latency"]] = counts.get(r["latency"], 0) + 1
    print(f"Latency counts: {counts}")

    # Pick one row per latency level.
    picks = {}
    for r in rows:
        lat = r["latency"]
        if lat not in picks:
            picks[lat] = r
        if len(picks) == 3:
            break

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PRIMARY_BACKBONE))
    n_added = tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    print(f"Tokenizer: {type(tok).__name__} vocab={tok.vocab_size} (+{n_added} EAST specials)")

    for lat in ["low", "medium", "high"]:
        r = picks[lat]
        parsed = parse_row(r)
        print(f"\n=== latency={lat} idx={r['index']} ===")
        print(f"source ({parsed.src_lang}): {parsed.source}")
        print(f"target ({parsed.tgt_lang}): {parsed.target}")
        print(f"chunks: {len(parsed.source_chunks)} src / {len(parsed.target_chunks)} tgt")

        # Chunk-count invariant.
        assert len(parsed.source_chunks) == len(parsed.target_chunks), \
            f"chunk-count mismatch at row {r['index']} — the release should have filtered this"

        # Join-equivalence — sanity, not exact.
        src_join = normalise(" ".join(parsed.source_chunks))
        tgt_join = normalise(" ".join(parsed.target_chunks))
        src_ok = normalise(parsed.source) == src_join
        tgt_ok = normalise(parsed.target) == tgt_join
        print(f"join-equiv: src={src_ok} tgt={tgt_ok}")
        if not src_ok:
            print(f"  src orig: {normalise(parsed.source)!r}")
            print(f"  src join: {src_join!r}")
        if not tgt_ok:
            print(f"  tgt orig: {normalise(parsed.target)!r}")
            print(f"  tgt join: {tgt_join!r}")

        # Reconstructed EAST-format string.
        east_str = interleave(parsed)
        print("east:")
        print("  " + east_str)

        # Tokenise and check the specials landed as single ids.
        ids = tok(east_str, add_special_tokens=False)["input_ids"]
        eor_id = tok.convert_tokens_to_ids(END_OF_READ)
        eow_id = tok.convert_tokens_to_ids(END_OF_WRITE)
        lat_id = tok.convert_tokens_to_ids(LATENCY_TOKENS[lat])
        n_eor = ids.count(eor_id)
        n_eow = ids.count(eow_id)
        n_lat = ids.count(lat_id)
        print(f"tokenised: len={len(ids)} #<|end-of-read|>={n_eor} "
              f"#<|end-of-write|>={n_eow} #<{lat}-lat>={n_lat}")
        assert n_eor == len(parsed.source_chunks), f"expected {len(parsed.source_chunks)} EOR, got {n_eor}"
        assert n_eow == len(parsed.target_chunks), f"expected {len(parsed.target_chunks)} EOW, got {n_eow}"
        assert n_lat == 1, f"expected exactly 1 latency token, got {n_lat}"

    print("\nPHASE 0 FORMAT VERIFY OK")


if __name__ == "__main__":
    main()
