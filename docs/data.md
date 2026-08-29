# Datasets

All data lives under `/g/data/po67/dipankar/data/simt-tor-26/`, accessed via the `data/` symlink in the repo root. Never committed.

**Currency note (2026-08-22):** the v6b multilingual pipeline uses 3 additional data assets not in the table below: (1) `results/phase2/multilingual_source_pool_v5.json` — 96K rows across 10 directions (8 covered after zh drop) sourced from SiMT-Multi-90K + custom Vi/Ar pools; (2) `results/phase2/annot_ot_multi_*/matrices.jsonl` — the 8 per-direction OT annotator matrices used by v6b builders; (3) **FLORES-200 devtest** at `/g/data/ba39/dipankar/simul-mt/data/raw/flores200/flores200_dataset/devtest/*.devtest` — 1012 sents × 200 languages, our primary streaming eval set for v6b (all BLEU/AL numbers in `05-phase2_sft_and_streaming.md` Phase 2b section).

## htgt pipeline (2026-08-24) — human-target training corpus

The v2bal_v3_htgt build uses a fully human-translated source pool for
de/ru pairs, replacing Multi-90K's GPT-4 target_chunks. Purpose: eliminate
GPT-4 dependency from BOTH annotation (already teacher-free via OT) AND
target translations. Gives the paper a clean "no GPT-4 anywhere" story.

**Source corpora (all human translations):**

| pair | corpus mix | rows | notes |
|---|---|---|---|
| de-en, en-de | europarl-v8 (78%) + news-commentary-v16 (11%) + TED2020 (11%) | 10K/dir | already on disk pre-htgt |
| ru-en, en-ru | TED2020 (58%) + news-commentary-v16 (42%) | 10K/dir | OPUS download 2026-08-24 (177195188) |
| ar-en, en-ar | TED2020 (100%) | 10K/dir | unchanged from v6b |
| vi-en, en-vi | TED2020 (100%) | 10K/dir | unchanged from v6b |

**Pooled at:** `results/phase2/multilingual_source_pool_htgt.json` (80K rows,
indexed 20000-139999 with no collisions — reindexed to preserve v6b ar/vi
matrices' original indices).

**Per-direction shards at:** `results/phase2/multilingual_source_pool_htgt_per_direction/{pair}.json`

**Known confound (2026-08-25):** htgt's de/ru sources come from europarl+NC+TED,
which is a DIFFERENT domain from CondB's de/ru sources (Multi-90K, derived
from WMT17-21 test data). This confounds the "GPT-4 target vs human target"
ablation with a source-domain change. A cleaner alternative — reuse
Multi-90K sources with WMT references — was investigated (see LOG.md
2026-08-25) but only 72% recoverable. Current htgt ships with the confound;
paper reframes as "trained on human-translated MT corpora" rather than
"same sentences as CondB but human targets".

## Multi-90K → WMT source-string recovery lookup (2026-08-25)

For the target-teacher-only ablation option (swap GPT-4 targets for WMT
human references while keeping identical sources), we recovered 72.6% of
Multi-90K de/ru source strings back to their original WMT test-set entries
via a 3-tier matching ladder (exact alnum → 8-token prefix jaccard≥0.75 →
5-token prefix jaccard≥0.85). See LOG.md 2026-08-25 for full method
and per-direction numbers.

**Artifact:** `results/phase2/m90k_wmt_recovery.pkl` (Python pickle).

Schema: `dict[direction: str, dict[m90k_source_string: str, wmt_reference: str]]`.

**Coverage:**
- de-en: 4,319 / 6,283 unique m90k srcs (68.7%)
- en-de: 4,793 / 6,186 (77.5%)
- ru-en: 4,064 / 5,625 (72.2%)
- en-ru: 5,871 / 8,153 (72.0%)

**WMT test sets consulted:** wmt13-wmt24 De/Ru bidirectional +
wmt14/full + wmt18/test-ts + wmt19/google/* variants. All under
`/g/data/ba39/dipankar/simul-mt/data/eval/{de-en,ru-en}/wmt{yy}*.{src,ref}`.

**sacrebleu cache** for downloads: `/g/data/po67/dipankar/cache/sacrebleu/`
(set via `SACREBLEU` env var; home dir was over quota).

## FLORES contamination in Multi-90K (2026-08-25)

**Multi-90K training data overlaps 40-76% with FLORES-200 devtest source
sentences per direction** (verified via 3-tier fuzzy matching: alnum-exact
→ 8-token-prefix jaccard≥0.75 → 5-token-prefix jaccard≥0.85). Any method
trained on Multi-90K — CondA, CondB (v2bal_v3), EAST-8B — has memorized a
huge fraction of the FLORES eval set.

**Composition of Multi-90K's ~5,000-8,000 unique sources per direction:**
- ~55% from WMT17-20 test sets (per EAST/ALMA claim)
- ~14% from FLORES dev + devtest (unexpected leakage)
- ~30% unknown provenance (probably additional WMT variants or
  paraphrased sentences)

**Implication for eval choice.** FLORES results for any Multi-90K-trained
method are inflated by memorization. Report primary numbers on WMT15,
WMT22, or IWSLT17 (all verified 0% contamination). FLORES becomes an
appendix showing the effect of training-data leakage.

**Our htgt corpus is verified clean** — 0/1012 overlap across all
direction × side × split combinations. FLORES numbers for
v2bal_v3_htgt are legitimate.

**Reproduce.** Script: same 3-tier matching as `m90k_wmt_recovery.pkl`
construction (LOG.md 2026-08-25). Applied to FLORES pairs instead of WMT.

| Asset | Path under `data/` | Rows | Role | Fetch |
|---|---|---|---|---|
| `SiMT-De-En-660K` | `SiMT-De-En-660K/SiMT-De-En-660K.json` | 660,876 | Primary SFT data. GPT-4-chunked at 3 latency levels. | `scripts/download_data.sh` (HF) |
| `SiMT-Multi-90K` | `SiMT-Multi-90K/` | 90,700 | Stretch — Stage II multilingual. **Also the source of cond-A-v6b's GPT-4 chunks** in the matched-backbone head-to-head (2026-08-22). | `scripts/download_data.sh` (HF) |
| **FLORES-200 devtest** | via `/g/data/ba39/dipankar/simul-mt/data/raw/flores200/` | 1012 × 200 langs | **Primary streaming eval set for v6b** (2026-08-22). | HF `openlanguagedata/flores_plus` |
| WMT15 De→En `newstest2015` | `wmt15-de-en/newstest2015.{de,en}` (also `.../simul-mt/data/eval/de-en/wmt15.*`) | 2,169 | EAST Fig. 3 head-to-head test set. | `sacrebleu -t wmt15 -l de-en` |
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
