# src/_archive/

Deprecated modules kept for provenance.

## `sft_pre_v6.py`
The pre-v6-pivot SFT recipe. Used base-model backbones (Gemma-4-E2B, not -it) and a latency-token vocabulary (`<low-latency>`, `<medium-latency>`, `<high-latency>`) prepended after BOS.

Superseded 2026-08-21 by the current `src/train/sft.py` (formerly `sft_v6.py`), which uses:
- **Instruct backbones** (`-it` variant) with chat templates.
- **Natural-language latency** injected into the user instruction (`"...with medium latency."`), not vocabulary tokens.
- Reduced special-token set: `SPECIAL_TOKENS_V6 = [EOR, EOW]`.

The pivot rationale is in `LOG.md` 2026-08-21 and `docs/hypotheses.md` §P1.

**Do not import from this file.** All live code paths use `src.train.sft`.
