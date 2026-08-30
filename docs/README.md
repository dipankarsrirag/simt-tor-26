# docs/ — reading order

Numbered. Read in order — later files assume earlier ones. Written so whoever picks up the project (Dipankar, next student, me next week) can see what was built, what was measured, and what's still to do.

| # | File | What it is | When to read |
|---|---|---|---|
| **00** | `00-README.md` | This file. Index + reading order. | Now. |
| **01** | `01-method_overview.md` | How the annotator works, mechanically. The commit criterion, monotonicity, chunk grouping, EAST interleave. | First — everything else assumes it. |
| **02** | `02-hypotheses.md` | Four core paper-facing hypotheses P1-P4: chunk-placement quality drives BLEU; robust across backbone/scale/language; OT-SFT is policy-agnostic partial translator; annotator is universal preprocessing. **Gate A** = P2 Qwen sub-claim; **Gate B** = P1 vs Simul-LLM published sub-claim. Consolidated 2026-08-18 late (H1-H23 archaeology deleted; git preserves it). | Second — every experiment traces to a hypothesis. |
| **03** | `03-phase1_annotator_experiments.md` | Phase 1 runs and results (Configs A → F/Gate 1). Which annotator setups we tried, what each found. Passes Gate 1 on OT at n=210 stratified. | Third — the empirical foundation for choosing OT + τ=0.30. |
| **04** | `04-random_floor_and_ot.md` | Intuition and worked examples for "random floor" (matched-chunk null Pearson) and OT with embedding-grounded ground cost. Read when tables in doc 03 mention "beats random" or you want to understand WHY OT catches things JS misses. | On demand. |
| **05** | `05-phase2_sft_and_streaming.md` | **The paper's headline result.** SFT pipeline (Gate 2), matched cond-A vs cond-B at n=10K, offline BLEU (32.4/32.5 — null rejected), streaming BLEU vs AL curve (cond-B +5 BLEU across wait-k). Bug diagnostics that matter. | Fourth — this is where "annotator paper" becomes "translation paper." |
| **06** | `06-data.md` | Datasets on disk: SiMT-660K, WMT13/15/22 test sets, RWTH gold alignments. Formats, fetch commands, EAST Eq. 4 for RWTH-A (Phase 3 appendix eval). | On demand — read when adding a new dataset or checking format. |
| **07** | `07-next_steps.md` | What to do next, in order. Blockers, gates, and sequencing. | Whenever picking up work. |

The single-source-of-authority files at the repo root are still:

- `../CLAUDE.md` — the project claim and non-negotiable invariants (kept lean; points here).
- `_archive/method-formal.md` — the annotation algorithm, precisely.
- `experiments.md` — the ablation grid, baselines, metrics.
- `_archive/TIMELINE.md` — phases and gates.
- **`../LOG.md`** — append-only run + decision log. The primary record; `docs/` is a curated summary of it.
- `setup.md` — compute, paths, accounts, ops rules.
- `_archive/OPTIONALS.md` — paper strategy: venue verdict, blockers, positioning.

## Naming — the live experimental arm (2026-08-22 update)

We now have **multiple trained arms** — the ship arm plus supporting head-to-head baselines re-introduced 2026-08-22 for a within-backbone matched comparison.

| Name | Path prefix | What it is |
|---|---|---|
| **v6b-ctrl-merged3** (ship) | `_archive/results/gemma_2b_curated/sft_multilingual_v6b_ctrl_merged3/final/` | Gemma-4-E2B-it + chat template + NL latency prompt + our OT-chunks with EAST §3.1 merge (<=3-word chunks folded forward). α=1 (no EOR/EOW upweight), 2 epochs, best-model-by-eval-loss, direct-ids splice. **The paper's headline model.** |
| **v6b-ctrl** (raw OT baseline) | `_archive/results/gemma_2b_curated/sft_multilingual_v6b_ctrl/final/` | Same recipe, no merge — raw OT chunks. Sits below merged3 by ~5 BLEU on average. |
| **v6b-ctrl-merged** (<2-word merge) | `_archive/results/gemma_2b_curated/sft_multilingual_v6b_ctrl_merged/final/` | EAST §3.1 original threshold. Middle of the pack. |
| **cond-A-v6b** (GPT-4 chunks head-to-head) | `_archive/results/gemma_2b_curated/sft_multilingual_v6b_condA/final/` | Matched-backbone baseline: SiMT-Multi-90K's GPT-4 chunks, same recipe as ship. 4 dirs only (de-en, en-de, ru-en, en-ru). |
| **v6b-ctrl-e4b** (scaling test) | `_archive/results/gemma_2b_curated/sft_multilingual_v6b_ctrl_e4b/final/` | Same v6b-ctrl training data on Gemma-4-**E4B**-it (4B). Notably **UNDERPERFORMS merged3** on E2B → chunk simplification beats scaling. |

**Deprecated arms (kept only for regression / LOG.md provenance):**
- `sft_multilingual_v6/final` — pre-v6b (pre-string-round-trip-fix, pre-α=1) with silent 40-47% AR/VI row drop; superseded 2026-08-22.
- `sft_multilingual_v6b/final` — α=5 EOR/EOW upweight variant; superseded by α=1 ctrl. Kept because LOG.md refers to the α=5 vs α=1 finding.

## Project state in one paragraph (as of 2026-08-22)

**Multilingual v6b-ctrl-merged3 is the ship model.** Gemma-4-E2B-it + chat-template + NL latency prompt (v6 recipe) + direct-ids splice training (no string round-trip, byte-identical annotator↔training↔inference) + α=1 (no special-token upweight; α=5 was hurting) + EAST §3.1 merge at <=3-word threshold. Trained on 79K rows across 8 language pairs (de-en, en-de, ar-en, en-ar, ru-en, en-ru, vi-en, en-vi). **N=50 FLORES devtest sanity (5 latencies × 8 dirs = 40 cells):** mean BLEU 29.46 for merged3 vs 24.89 raw OT ctrl vs 30.51 cond-A (GPT-4 chunks, 4 dirs only). **On de-en at low_medium latency merged3 (31.88) beats cond-A (30.90)** — we beat GPT-4 chunks on our own backbone. **E4B scaling test underperformed merged3 (E2B) by −0.49 BLEU** → chunk simplification is a bigger lever than doubling model size for this problem. Big finding this cycle: the +5.72 BLEU gap between raw OT and GPT-4 chunks is 76% recoverable via EAST-style chunk merging (see LOG.md `[DECISION] 2026-08-22 — v6b-ctrl-merged3 is the new ship candidate`). Full N=1012 FLORES + WMT15 numbers are the next milestone. Other closed decisions this cycle: v6 pivot (2026-08-21, instruct backbone + chat template); v6b string round-trip fix (2026-08-22); DAL as primary latency metric alongside AL/LAAL. Target venue: ACL/EMNLP Findings or IWSLT.

## The five biggest bugs caught this project (paste-worthy for a lessons-learned)

1. **Prompt confound (Phase 1)** — running `gemma-4-E2B-it` with raw concat made JS look degenerate. Switching to base Gemma with raw concat (matching METHOD §1's spec exactly) is what unlocked the whole method. See H2/H3 in `02-hypotheses.md`.

2. **Embedding-init collapse (Phase 2 SFT)** — my first pass initialized all 5 new EAST-token embedding rows to the SAME mean-of-existing value. LM head then couldn't distinguish them (symmetric loss landscape). Special-token loss 11.87 nats. 0/30 probes emitted tags. Fixed by using transformers' default mean-covariance random init. See doc 05.

3. **Extrinsic offline gen didn't stop at `<|end-of-write|>`** (Phase 2 eval) — cond-A never saw a "one giant chunk" training row, so after emitting a target it kept producing more `src <eor> tgt <eow>`. hyp/ref length 1.99, BLEU depressed to 15.89. Post-fix (add `<|end-of-write|>` as an `eos_token_id`): BLEU 32.41.

4. **`sft.py --corpus_file` capped at --n_sentences default** — cond-B first training silently used 2K of 9,567 rows. `n_rows_trained` in `sft_summary.json` caught it. Every field in a config dump should be inspected once.

5. **`MAX_SHARDS` chain-at-start gate hit before annotation finished** — the E4B cond-B annotation shard chain (`jobs/phase2_annot_ot_e4b_n10k_shard.pbs`) capped at `MAX_SHARDS=15` which was set from a first-cut wall-time estimate. Annotation stalled at 26% completion — the ceiling was silently reached, no further shards queued, and I only noticed because I checked `qstat` after ~24h of nothing being scheduled. Fix (commit `42ab554`): bumped E4B ceiling to 40 and Qwen ceiling to 25. **Any resource ceiling written in a job script is an assumption that will break.** Log the "shard N of MAX_SHARDS" line prominently; check the numerator against the denominator when reviewing progress.

Common thread: each bug SILENTLY produced wrong numbers (or silently STOPPED producing them, in bug 5's case) rather than crashing. The fix in each case took one line; the diagnosis took hours. **Log per-token/per-field details AND per-shard progress markers, not just aggregates.**
