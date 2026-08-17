# docs/ — reading order

Numbered. Read in order — later files assume earlier ones. Written so whoever picks up the project (Dipankar, next student, me next week) can see what was built, what was measured, and what's still to do.

| # | File | What it is | When to read |
|---|---|---|---|
| **00** | `00-README.md` | This file. Index + reading order. | Now. |
| **01** | `01-method_overview.md` | How the annotator works, mechanically. The commit criterion, monotonicity, chunk grouping, EAST interleave. | First — everything else assumes it. |
| **02** | `02-hypotheses.md` | Falsifiable hypotheses H1-H9. Rationale, prediction, test, outcome per hypothesis. H1-H7 are Phase 1 (annotator); H8-H9 are Phase 2 (SFT + streaming). | Second — every experiment traces to a hypothesis. |
| **03** | `03-phase1_annotator_experiments.md` | Phase 1 runs and results (Configs A → F/Gate 1). Which annotator setups we tried, what each found. Passes Gate 1 on OT at n=210 stratified. | Third — the empirical foundation for choosing OT + τ=0.30. |
| **04** | `04-random_floor_and_ot.md` | Intuition and worked examples for "random floor" (matched-chunk null Pearson) and OT with embedding-grounded ground cost. Read when tables in doc 03 mention "beats random" or you want to understand WHY OT catches things JS misses. | On demand. |
| **05** | `05-phase2_sft_and_streaming.md` | **The paper's headline result.** SFT pipeline (Gate 2), matched cond-A vs cond-B at n=10K, offline BLEU (32.4/32.5 — null rejected), streaming BLEU vs AL curve (cond-B +5 BLEU across wait-k). Bug diagnostics that matter. | Fourth — this is where "annotator paper" becomes "translation paper." |
| **06** | `06-data.md` | Datasets on disk: SiMT-660K, WMT13/15/22 test sets, RWTH gold alignments. Formats, fetch commands, EAST Eq. 4 for RWTH-A (Phase 3 appendix eval). | On demand — read when adding a new dataset or checking format. |
| **07** | `07-next_steps.md` | What to do next, in order. Blockers, gates, and sequencing. | Whenever picking up work. |

The single-source-of-authority files at the repo root are still:

- `../CLAUDE.md` — the project claim and non-negotiable invariants (kept lean; points here).
- `../METHOD.md` — the annotation algorithm, precisely.
- `../EXPERIMENTS.md` — the ablation grid, baselines, metrics.
- `../TIMELINE.md` — phases and gates.
- **`../LOG.md`** — append-only run + decision log. The primary record; `docs/` is a curated summary of it.
- `../HOUSEKEEPING.md` — compute, paths, accounts, ops rules.
- `../OPTIONALS.md` — paper strategy: venue verdict, blockers, positioning.

## Project state in one paragraph (as of 2026-08-18)

Phase 1 (annotator): DONE. Chose base Gemma-4-E2B + raw concat + OT (τ=0.30). Gate 1 passed on stratified n=210. Phase 2 (SFT): matched cond-A (GPT-4 chunks) vs cond-B (OT chunks) trained on same 9,567 sentences from SiMT-660K. Offline BLEU: 32.4 / 32.5 — null rejected, cond-B doesn't degrade translation. **Streaming BLEU: cond-B beats cond-A by +4.8-5.7 BLEU across wait_k ∈ {3, 5, 7} at matched AL** — the paper's headline. Cross-family (Qwen3.5-2B) and scale (Gemma-4-E4B base) replications in flight. Data-scale curve (10K → 50K on champion model) queued. See `05-phase2_sft_and_streaming.md` for the story and `07-next_steps.md` for what's next.

## The four biggest bugs caught this project (paste-worthy for a lessons-learned)

1. **Prompt confound (Phase 1)** — running `gemma-4-E2B-it` with raw concat made JS look degenerate. Switching to base Gemma with raw concat (matching METHOD §1's spec exactly) is what unlocked the whole method. See H2/H3 in `02-hypotheses.md`.

2. **Embedding-init collapse (Phase 2 SFT)** — my first pass initialized all 5 new EAST-token embedding rows to the SAME mean-of-existing value. LM head then couldn't distinguish them (symmetric loss landscape). Special-token loss 11.87 nats. 0/30 probes emitted tags. Fixed by using transformers' default mean-covariance random init. See doc 05.

3. **Extrinsic offline gen didn't stop at `<|end-of-write|>`** (Phase 2 eval) — cond-A never saw a "one giant chunk" training row, so after emitting a target it kept producing more `src <eor> tgt <eow>`. hyp/ref length 1.99, BLEU depressed to 15.89. Post-fix (add `<|end-of-write|>` as an `eos_token_id`): BLEU 32.41.

4. **`sft.py --corpus_file` capped at --n_sentences default** — cond-B first training silently used 2K of 9,567 rows. `n_rows_trained` in `sft_summary.json` caught it. Every field in a config dump should be inspected once.

Common thread: each bug SILENTLY produced wrong numbers rather than crashing. The fix in each case took one line; the diagnosis took hours. **Log per-token/per-field details, not just aggregates.**
