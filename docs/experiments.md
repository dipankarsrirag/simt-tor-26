# Phase 1 experiments and results

The runs, in order, with what each tested and what it found. Cross-references `hypotheses.md`.

Every run also has an entry in `../LOG.md`; this document is the readable summary.

## Setup common to all runs

- **Backbone family:** Gemma-4-E2B (both `-it` and base variants), on `/g/data/po67/dipankar/models/gemma-4-{E2B, E2B-it}/`.
- **Data:** `SiMT-De-En-660K` (WMT15-derived; GPT-4-chunked; 660,876 rows across low/medium/high latency).
- **Sample:** 48 sentences, balanced across the three latency levels (`seed=42`, max source token count 80 after tokenisation).
- **Compute:** 1× H200 (`gpuhopper` queue), typically 30 min walltime. Model load ~30–50 s; annotation ~1.3 s/sentence for JS/KL, ~5–10× slower for OT.
- **Analyses run offline on each `matrices.jsonl`:**
  - `phase1_random_floor.py` — JS vs uniform-random-at-matched-chunk-count.
  - `phase1_entropy_sweep.py` — entropy-only criterion (H4).
  - `phase1_gpt4_pearson.py` — GPT-4 baseline from the shipped chunks.
  - `phase1_per_sentence_compare.py` — matched-chunk-count comparison, r-of-Pearsons, reordering candidates.

## Config A — `-it` + raw concat + JS  ↔ tests H1

**Job:** `176267898.gadi-pbs`. Results at `results/phase1_tau_sweep/`.

**Sweep:**
| τ | fire% | ours_ch | GPT-4 ch | Pearson med | Pearson min |
|---|---|---|---|---|---|
| 0.02 | 8% | 1.10 | 4.06 | 0.33 | 0.25 |
| 0.05 | 23% | 1.67 | 4.06 | 0.53 | 0.28 |
| 0.10 | 52% | 3.31 | 4.06 | 0.82 | 0.22 |
| 0.15 | 69% | 4.19 | 4.06 | 0.86 | 0.00 |
| 0.20 | 77% | 5.92 | 4.06 | 0.92 | 0.00 |
| 0.30 | 94% | 8.58 | 4.06 | 0.96 | 0.42 |

**Random floor.** JS Pearson_med > random Pearson_med at EVERY tau (JS worse than uniform-random-monotone-at-matched-chunk-count by 2–15 pp). *Apparent-confirms H1 that JS is degenerate on this backbone.*

**Overturned.** The advisor caught a confound before we spent more compute: `gemma-4-E2B-it` is instruction-tuned; under raw concat it doesn't do translation. Move to H2.

## Config B — `-it` + chat template + JS  ↔ tests H2

**Job:** `176272966.gadi-pbs`. Results at `results/phase1_tau_sweep_chat/`.

**Sweep:**
| τ | fire% | ours_ch | Pearson med | Pearson min |
|---|---|---|---|---|
| 0.02 | 100% | 7.19 | 0.93 | 0.49 |
| 0.05 | 100% | 8.02 | 0.94 | 0.56 |
| 0.10 | 100% | 9.10 | 0.95 | 0.60 |
| 0.15 | 100% | 9.60 | 0.96 | 0.55 |
| 0.20 | 100% | 9.73 | 0.96 | 0.55 |
| 0.30 | 100% | 10.02 | 0.97 | 0.78 |

**Random floor.** Gap narrowed from ~15 pp (raw) to ~2 pp (chat) — JS still barely loses at every tau, but closer to signal.

**GPT-4 baseline (same 48 sentences, computed here for the first time):**
- **GPT-4 Pearson_med = 0.943.** min = 0.693, max = 0.984. Mean chunks/sentence = 4.06.
- **Read:** WMT De→En at 30–50 tokens (after EAST App. C monotonicity filter) is inherently diagonal. Ours matching GPT-4 in aggregate isn't degeneracy — it's the data.

**Per-sentence comparison at per-sentence matched-count tau:**
- Ours Pearson_med = 0.919, min = 0.313. Chunks_mean = 5.98 (vs GPT-4's 4.06 — even the strictest tau=0.01 gives finer chunks than GPT-4 on some sentences).
- **Per-sentence r(GPT-4, ours) = 0.149.**

**Reordering candidates (lowest GPT-4 Pearson):** 5 of 8 MATCH (ours also Pearson < 0.85), 3 MISS. Notable MISS: idx=553850 — GPT-4 gives 2 chunks committing at i=42/53 (Pearson 0.693) on a German subject-verb-split; ours gives 7 chunks (Pearson 0.907) committing early on the cognate prefix.

**Read.** Chat template fixed the fire rate but not the per-sentence discordance with GPT-4. Move to H3.

## Config C ★ — base + raw concat + JS  ↔ tests H3 and H4

**Job:** `176304944.gadi-pbs`. Results at `results/phase1_tau_sweep_base/`.

**Sweep:**
| τ | fire% | ours_ch | Pearson med | Pearson min | JS beats random? |
|---|---|---|---|---|---|
| 0.02 | 6% | 1.12 | 0.39 | 0.30 | no |
| 0.05 | 52% | 2.19 | 0.33 | 0.00 | no |
| 0.10 | 79% | **3.46** | **0.73** | 0.00 | no |
| **0.15** | 92% | 6.04 | **0.84** | 0.00 | **YES** (JS 0.842 < random 0.881) |
| 0.20 | 94% | 7.62 | 0.94 | 0.00 | no |
| 0.30 | 98% | 10.04 | 0.97 | 0.78 | no |

**First observation of JS beating random-at-matched-latency** — at τ=0.15, JS is less diagonal than uniform-random-monotone with the same chunk count.

**Per-sentence comparison:**
- Ours Pearson_med = **0.778**, chunks_mean = 2.96 (vs GPT-4's 4.06 — closer than under -it+chat).
- Chunk-count delta mean_abs = 1.44 (vs -it+chat's 2.25).
- **Per-sentence r(GPT-4, ours) = 0.175** (barely improved from -it+chat's 0.149, but see the reordering catch).

**Reordering catch on idx=553850 (the walked MISS from Config B):**
- GPT-4: 2 chunks, Pearson=0.693 — reads to i=42 before committing.
- **Ours (base+raw, matched-count tau): 2 chunks, Pearson=0.311.** ✓ Matches GPT-4's late-commit pattern.
- (Under Config B it was 7 chunks Pearson=0.907 — a clear MISS.)

**Entropy-only sweep on the base matrices (H4 test):**
| H_tau | fire% | ours_ch | Pearson med |
|---|---|---|---|
| 0.5 | 27% | 1.29 | 0.33 |
| 1.0 | 50% | 1.60 | 0.37 |
| 2.0 | 79% | **3.50** ≈ GPT-4 | 0.83 |
| 3.0 | 98% | 5.50 | 0.90 |
| 4.0 | 100% | 5.00 | 0.91 |

At matched chunk count (H_tau=2.0: 3.50 chunks; JS τ=0.10: 3.46 chunks), JS gives Pearson_med 0.73 vs entropy-only's 0.83 — a 10 pp gap in JS's favour. Suggests `P_full` is doing work (H4 provisionally supported), but the H_tau grid is coarse; needs finer sweep for a clean verdict.

**Read.** Base+raw is materially better than -it+chat: chunk counts land near GPT-4, catches individual reordering cases, JS beats random at τ=0.15 (first time). The per-sentence r stays low because monotonic-majority-sentence-Pearson variance dominates the aggregate — the r-metric isn't the right primary signal.

## Config D ★★ — base + raw concat + OT  ↔ tests H5

**Job:** `176307323.gadi-pbs`. Results at `results/phase1_tau_sweep_ot/`. Completed: 25 min annotation (31s/sentence — ~24× slower than JS, expected due to Sinkhorn iterations).

**Sweep:**
| τ | fire% | ours_ch | Pearson med | Pearson min |
|---|---|---|---|---|
| 0.02 | 0% | 1.00 | — | — |
| 0.05 | 0% | 1.00 | — | — |
| 0.10 | 10% | 1.15 | 0.30 | 0.00 |
| 0.15 | 48% | 1.85 | 0.30 | 0.00 |
| 0.20 | 71% | 2.69 | 0.63 | 0.00 |
| **0.30** | 90% | **4.67 ≈ GPT-4** | **0.81** | 0.00 |
| 0.50 | 98% | 9.04 | 0.96 | 0.63 |

**Random floor.** OT beats random-at-matched-chunk-count at TWO tau values (0.20 and 0.30), vs JS which beat random at only one (0.15).

| τ | OT_med | RD_med | OT beats RD? |
|---|---|---|---|
| 0.20 | 0.625 | 0.685 | **YES** |
| 0.30 | 0.805 | 0.841 | **YES** |
| 0.50 | 0.960 | 0.940 | no |

**Per-sentence comparison (matched-chunk-count tau_ot per sentence):**
- **Per-sentence r(GPT-4, OT-ours) = 0.306** (n=37; the other 11 sentences collapsed to single-chunk under OT, Pearson undefined).
- Up from JS's 0.175 with n=48. Meaningful improvement.
- Ours chunks_mean = 3.27 (vs GPT-4's 4.06). Chunk-count delta mean_abs = 1.42.

**Reordering candidates (top 8 lowest GPT-4 Pearson):** 3 MATCH, 5 MISS. But of the 5 MISS: 4 are single-chunk collapse (OT never fires ≤ τ=0.50 on those sentences), not "wrong" — they're conservatively over-committed.
- idx=553850 (walked verb-final): MATCH (Pearson=0.311, 2 chunks). Same catch as JS Config C.
- idx=493988: MATCH — OT gives Pearson=0.663, better than JS Config C's 0.808.
- idx=555138, 359904, 537446, 367208: single-chunk under OT at τ ≤ 0.50 (coverage limit).

**Coverage caveat (RESOLVED by Config D-ext).** See below.

## Config D-ext ★★★ — same OT, extended τ grid

**Job:** `176318744.gadi-pbs`. Results at `results/phase1_tau_sweep_ot_ext/`. τ grid extended to `{0.30, 0.50, 0.70, 1.00, 1.30}` to close the single-chunk collapses on the 4 hard sentences.

**Sweep:**
| τ | fire% | ours_ch | Pearson med | Pearson min |
|---|---|---|---|---|
| 0.30 | 90% | 4.67 | 0.81 | 0.00 |
| 0.50 | 98% | 9.04 | 0.96 | 0.63 |
| **0.70** | **100%** | 6.73 | 0.93 | **0.34** |
| 1.00 | 100% | 1.02 | ~0 | 0.00 |
| 1.30 | 100% | 1.00 | ~0 | 0.00 |

τ=0.70 gives 100% fire with Pearson_min=0.34 — best min we've observed at any tau of any config. τ ≥ 1.00 collapses: the criterion fires at i=1 for all target tokens, one giant chunk.

**Per-sentence comparison (matched-chunk-count tau_ot per sentence, grid now `{0.30, ..., 1.00}`):**
- **r(GPT-4, ours) = 0.222**, n=47 defined (was 0.306, n=37 under narrow grid — new sentences with imperfect matches lower r but include more reordering candidates).
- Ours chunks_mean = **3.98** (vs GPT-4's 4.06 — essentially matched).
- **Chunk-count delta mean_abs = 0.62** (was 1.42 — dramatically closer to GPT-4).
- Ours Pearson_med = 0.854.

**Reordering catches (top-8 lowest GPT-4 Pearson): 6 MATCH, 2 MISS.**

| idx | GPT-4 P | Config C (JS) | Config D (OT ≤0.50) | **Config D-ext (OT ≤1.00)** |
|-----|---------|---------------|---------------------|------------------------------|
| 553850 | 0.693 | MATCH 0.311 | MATCH 0.311 | MATCH 0.835 |
| 555138 | 0.765 | MISS (NaN) | MISS (NaN) | MISS 0.870 |
| 596095 | 0.798 | MATCH 0.846 | MATCH 0.845 | MATCH 0.845 |
| 493988 | 0.826 | MATCH 0.808 | MATCH 0.661 | MATCH 0.661 |
| 502711 | 0.834 | MISS 0.916 | MISS 0.866 | MISS 0.866 |
| 359904 | 0.839 | MATCH 0.846 | MISS (NaN) | **MATCH 0.751** ★ |
| 537446 | 0.860 | MISS 0.910 | MISS (NaN) | **MATCH 0.340** ★★ |
| 367208 | 0.863 | MATCH 0.489 | MISS (NaN) | **MATCH 0.847** ★ |

**Read.** With the extended τ grid, OT MATCHES 6/8 reordering candidates — the best any config has done. idx=537446 gives Pearson=0.34, the lowest per-sentence Pearson achieved anywhere. The two remaining MISSes (555138, 502711) are close to threshold (0.87, 0.87) — a per-sentence-Pearson threshold of 0.87 instead of 0.85 would count them too.

## Table: all configs side by side (final)

Matched to each config's most-informative τ (chunk-count closest to GPT-4's 4.06 or the "beats random" τ where present).

| Config | Model / Prompt / Crit | τ | Fire % | ours_ch | Chunks Δ | Reord MATCH | r(GPT-4, ours) |
|---|---|---|---|---|---|---|---|
| A | -it, raw, JS | 0.10 | 52% | 3.31 | 0.75 | — | — |
| B | -it, chat, JS | 0.10 | 100% | 9.10 | 5.04 | 5/8 | 0.149 |
| C | base, raw, JS | 0.10 | 79% | 3.46 | 0.60 | 5/8 | 0.175 |
| D | base, raw, OT (τ≤0.50) | 0.30 | 90% | 4.67 | 0.61 | 3/8 (4 NaN) | 0.306 |
| **D-ext ★★★** | **base, raw, OT (τ≤1.00)** | **per-sent 0.30-1.00** | **100%** | **3.98** | **0.62** | **6/8** | **0.222** |

**Read.**
- H5 SUPPORTED: OT catches more signal than JS by two independent metrics — wider "beats random" range, higher per-sentence correlation with GPT-4.
- The paper's headline stands up: the embedding-grounded ground cost earns its keep.
- Cost is real (~24× JS). Justifiable per METHOD §3 as the primary criterion; JS remains the cheap-baseline ablation.

## Table: all configs side by side

Matched to each config's most-informative τ (chunk-count closest to GPT-4's 4.06 or the "beats random" τ where present).

| Config | Model / Prompt / Crit | τ | Fire % | ours_ch | Pearson_med | Beats random? | r(GPT-4, ours) |
|---|---|---|---|---|---|---|---|
| A | -it, raw, JS | 0.10 | 52% | 3.31 | 0.82 | no | — |
| B | -it, chat, JS | 0.10 | 100% | 9.10 | 0.95 | no | 0.149 |
| C | base, raw, JS | 0.10 | 79% | 3.46 | 0.73 | (yes @ 0.15) | 0.175 |
| **D ★★** | **base, raw, OT** | **0.30** | **90%** | **4.67** | **0.81** | **yes @ 0.20 & 0.30** | **0.306** |

## Config F ★★★ (Gate 1) — n=210 stratified-by-reordering, base + raw + {OT, JS}

**Jobs:** `176387597` (OT, walltime 1:38:53), `176387598` (JS, walltime 0:06:16). Full report: `results/gate1/gate1_report.md`.

**Sample.** 210 sentences stratified by GPT-4 per-sentence Pearson using fixed absolute thresholds (`monotone ≥ 0.90`, `mild 0.70-0.90`, `reordering < 0.70`), 70 per bin, seed 42. Precomputed on the full 660K corpus (631,915 rows after 80-tok filter) — bin distribution across the corpus is 74.3% / 24.4% / 0.7% / 0.7% undefined, so the reordering bin is genuinely rare and worth stratifying explicitly.

**Metric.** Per-sentence matched-chunk-count τ. Coverage = fraction of bin where trace has > 1 chunk. **Effective MATCH%** = (covered ∧ ours_pearson < 0.85) / bin_n — the honest number for the mechanism claim, treating single-chunk collapse as MISS.

| Criterion | Bin | n | Coverage | Δ chunks | GPT-4 pear | Ours pear | **MATCH_eff** |
|---|---|---|---|---|---|---|---|
| **OT (winning)** | monotone | 70 | 100% | 0.67 | 0.949 | 0.827 | **38.6%** |
| **OT** | mild | 70 | 77.1% | 0.43 | 0.846 | 0.605 | **60.0%** |
| **OT** | reordering | 70 | 77.1% | 0.46 | 0.645 | 0.640 | **54.3%** |
| JS (ablation) | monotone | 70 | 80.0% | 1.60 | 0.949 | 0.701 | 55.7% |
| JS | mild | 70 | 48.6% | 0.80 | 0.846 | 0.553 | 44.3% |
| JS | reordering | 70 | 45.7% | 0.64 | 0.645 | 0.408 | 44.3% |

**Read — Gate 1 PASSES for OT.**

- **Monotone bin:** OT matches GPT-4 chunk-count tightly (Δ = 0.67) at 100% coverage. Pass.
- **Reordering bin:** OT effective MATCH 54.3% strictly beats monotone 38.6% by 15.7 pp — the mechanism claim (margin widens on reordering-divergent pairs) confirmed at n=210. Coverage 77.1% above the 70% threshold in `TIMELINE.md` Gate 1. Pass.
- **Mild bin:** effective MATCH 60.0% — the highest of any bin, with the tightest chunk-count Δ. The middle-ground non-monotone sentences where OT's embedding-grounded ground cost has the most to bite on.
- METHOD §8 sanity: positional Pearson median 0.78 (mono 0.88, mild 0.73, reord 0.73); zero identity-like traces; 5.7% terminal-degenerate. Criterion is non-degenerate. See `results/gate1/gate1_report.md`.

**Bin ordering caveat.** The `CLAUDE.md` mechanism claim predicts `monotone < mild < reordering`. Actual is `monotone ≪ {reordering ≈ mild}` (38.6 < 54.3 < 60.0). Conditional MATCH is nearly-monotone across bins (mono 38.6 < reord 70.4 ≈ mild 77.8); the mild-vs-reordering gap opens because 16% of reordering-bin sentences remain single-chunk-collapse even under OT — the true late-commit-required tail (see walked examples in `gate1_report.md`). The paper's framing should be "bimodal-vs-monotone", not "monotonically widening margin".

**Read — Gate 1 FAILS for JS.** No mechanism concentration — effective MATCH is 56% monotone / 44% mild / 44% reordering, no lift on the reordering bins. JS collapses to single-chunk on 54% of the reordering bin because at strict τ it does not fire when P_pre and P_full concentrate on different (but semantically similar) tokens. **This rules out the "ship JS with a shorter method section" branch** — the paper's headline criterion is OT.

**Caveat.** Without gold alignment, agreement-with-GPT-4 is *not* tag quality — this is a gate, not a paper result. The RWTH-A intrinsic (EAST App. E.4 mirror) runs in Phase 3 and is not skipped.

**Unlocks:** Phase 2 SFT. Annotate 10K/50K with OT-winning config, matched-condition training (A = GPT-4 tags, B = ours), WMT15 newstest2015 extrinsic eval with BLEU/COMET/BLEURT vs AL/LAAL/**AL-CA**.

## Phase 2 — SFT pipeline (in flight)

Full run details in `LOG.md`; this section is the readable summary.

**Pipeline validated end-to-end.** trl.SFTTrainer 1.10 + Gemma-4-E2B base +
extended tokenizer (5 EAST tokens at ids 262144–262148, `results/phase2/tokenizer-extended/`)
+ full-sequence CE loss per EAST §3.2 + mean-covariance embedding init. See
LOG.md 2026-08-16 for the load-bearing embedding-init bug fix.

**Gate 2 PASSES (cond-A, 2K/3e).** 30/30 streaming probes emit `<|eor|>` + `<|eow|>`
in correct alternation.

**Cond-A n=10K trained (job 176432676).** Early stopping fires at step 650
(epoch 1.14); best eval_loss 1.613 at checkpoint-500. 40/40 streaming probes
emit EOR+EOW cleanly under 3-word-prefix + latency-token prompts.

**Batched OT annotator (2026-08-17).** `ot_divergence_row_batched` — log-domain
Sinkhorn across all m target positions, one GPU-saturating call per source-prefix
length. 14× speedup on H200 (2s/sent vs 28s/sent) with L∞ diff 7e-6 vs the
per-pair reference impl on a CPU smoke test. Cond-B n=10K annotation now
compute-feasible in <8h wall (was multi-day).

**Cond-B n=10K annotation (in flight, ~76% done at last check).** OT criterion,
τ grid `{0.30, 0.50, 0.70, 1.00}`, same 9,567 indices as cond-A n=10K. Chained
self-resubmitting 1h shards via `qsub -W depend=afterany` at wrapper START
(see LOG.md 2026-08-17 decision — prevents PBS SIGKILL from breaking the resume
chain).

**Extrinsic eval harness (scaffolded, `src/eval/extrinsic.py`).** Layer 1
(offline BLEU) implemented. Layers 2–3 (streaming state machine + AL/AL-CA)
pending Layer 1 sanity numbers on newstest2013 dev. WMT13 De→En dev set fetched
via sacrebleu at `/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/`.
