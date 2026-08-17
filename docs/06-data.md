# Datasets

All data lives under `/g/data/po67/dipankar/data/simt-tor-26/`, accessed via the `data/` symlink in the repo root. Never committed.

| Asset | Path under `data/` | Rows | Role | Fetch |
|---|---|---|---|---|
| `SiMT-De-En-660K` | `SiMT-De-En-660K/SiMT-De-En-660K.json` | 660,876 | Primary SFT data. GPT-4-chunked at 3 latency levels. | `scripts/download_data.sh` (HF) |
| `SiMT-Multi-90K` | `SiMT-Multi-90K/` | 90,700 | Stretch — Stage II multilingual. | `scripts/download_data.sh` (HF) |
| WMT15 De→En `newstest2015` | `wmt15-de-en/newstest2015.{de,en}` | 2,169 | Primary SiMT test (EAST Fig. 3). | `sacrebleu -t wmt15 -l de-en` |
| WMT22 8 pairs | `wmt22/*/newstest2022.*` | ~2K each | Stretch multilingual + document-level test. | `sacrebleu -t wmt22` |
| **RWTH De→En gold alignments** | `rwth-de-en/DeEn/` | **509** | **Phase 3 appendix eval (EAST App. E.4). Deferred from Gate 1 per 2026-08-16 decision.** | **Manual browser step, see below.** |

## SiMT-De-En-660K format

JSON list of dicts (one per parallel pair):
```json
{
  "index": 0,
  "source": "Eine bedeutend höhere Sparquote ...",
  "target": "A significantly higher saving rate ...",
  "src_lang": "German",
  "tgt_lang": "English",
  "latency": "low",         // "low" / "medium" / "high"
  "source_chunks": ["Eine bedeutend höhere Sparquote", ...],
  "target_chunks": ["A significantly higher saving rate", ...]
}
```

Chunk counts always match between source and target (EAST App. C discards mismatched rows before release). Whitespace-join equivalence to `source`/`target` verified in Phase 0 (`scripts/phase0_verify_east_format.py`).

Latency distribution: low=230,902 / medium=227,131 / high=202,843.

## RWTH gold alignments — Phase 3 appendix eval (deferred from Gate 1)

The 2026-08-16 decision (`LOG.md`) redefined Gate 1 to a stratified-by-reordering aggregate on 200 SiMT-660K sentences and moved RWTH-A to the Phase 3 appendix, mirroring EAST's App. E.4 positioning. The dataset below is unchanged; only when it runs is different.

**URL:** `https://www-i6.informatik.rwth-aachen.de/goldAlignment/`
**Dataset:** "Gold Alignment for Europarl German-English Dataset", v1.0.
**Access:** browser registration form (name / organisation / email + non-commercial licence acceptance). Not scriptable.
**Licence:** free for non-commercial use; may not be redistributed.

**Files inside `data/rwth-de-en/DeEn/`:**
| File | Lines | Format |
|---|---|---|
| `de` | 509 | One De sentence per line. **Latin-1 encoded** (`daß` shows as `da�` under UTF-8) — reencode when reading. |
| `en` | 509 | One En sentence per line, UTF-8. |
| `alignmentDeEn` | 11,551 | Long-form: SENT markers, `S i j` (sure) / `P i j` (possible) entries. |
| `alignmentDeEn.talp` | 509 | Short-form talp: one line per sentence, entries like `3-4 5-5 2p3 25p22 ...`. `i-j` = sure alignment src word `i` → tgt word `j` (1-indexed); `ipj` = possible alignment. |

**Example.** Sentence 0:
- De: `Wir glauben nicht , daß wir nur Rosinen herauspicken sollten .` (11 words)
- En: `We do not believe that we should cherry-pick .` (9 words)
- talp: `9-8 8-8 7-8 6-6 1-1 2-2 4-5 5-5 3-3 11-9 10-7 2-4`
  - `9-8` = source word `herauspicken` aligns to target word `cherry-pick`.
  - `10-7` = source word `sollten` aligns to target word `should` — reordering (`sollten` is source pos 10, `should` is target pos 7).

**Word-level vs token-level.** Alignments are over whitespace-separated words; our commits are over sub-word tokens. The intrinsic-eval script has to map word alignment → token alignment (Gemma's `GemmaTokenizer` sub-word segmentation applied to the whitespace-tokenised words).

## The Gate-1 metric — EAST Eq. 4

Following Zhang and Feng, 2022. From EAST §E.4:

> For a target token `y_i` with ground-truth aligned source position `a_i`, the number of source tokens read at generation time `g_i` must be at least equal to `a_i` for the alignment to be satisfied.
>
> `A = (1/T) Σ_{i=1..T} I[a_i ≤ g_i]`
>
> where `T` is the total number of target tokens.

Reads: "fraction of target tokens where at least the gold-aligned source position had already been read by commit time." Higher = more faithful. A policy that always waits for the whole source has `A = 1` (but very high latency).

**Ambiguity to resolve when implementing.** A target word can have multiple source alignments (multi-alignment); does `a_i` = min or max of the aligned positions? EAST doesn't specify explicitly; the safest reading is `a_i = max(aligned source positions)` (must have read the furthest-aligned source word). Also: do we count `P`-marked possible alignments, or only `S`-sure ones? Convention in the wait-k / SiMT literature is to use only `S`. Confirm on first implementation and log the choice.

## What we do and don't touch on each dataset

- **SiMT-660K:** annotate. This is the SFT training-data source. **Not a test set** — free to hyperparameter-search on.
- **WMT15 newstest2015:** the primary sentence-level SiMT test. Never annotate at inference; never select `tau` on it.
- **WMT22:** stretch test set for multilingual and doc-level. Not touched in Phase 1.
- **RWTH:** intrinsic annotation-quality eval only. Score both ours' and GPT-4's tags against the same 509 sentences.

## Sanity checks to run when a new dataset lands

1. `du -sh` and line counts against a known-good source (HF page or paper's Table).
2. Encoding — try UTF-8 first, fall back to Latin-1 (RWTH's `de` file needs Latin-1).
3. A small `docs/data/<name>.md` note recording: source URL, download date, sha256 of the raw archive, any preprocessing. `HOUSEKEEPING §3` rule.

RWTH sha256 (as fetched): `5aea49f44a9da4cf575d2dd303a8e12ebe7ba8b615ede7c28e7f8b0a0eb95793 DeEnGoldAlignment.tar.gz`.
