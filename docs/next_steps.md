# Next steps, in order

Ordered by priority. Each item states what it does, why now, and what unlocks after.

## 1. Gate 1 — stratified-by-reordering aggregate on 200 SiMT-660K sentences (redefined 2026-08-16)

Replaces the prior RWTH-based Gate 1 (see `LOG.md` 2026-08-16 decision). This is a *gate*, not a paper result — passing greenlights Phase 2; RWTH intrinsic A-score runs in Phase 3 as the paper's App. E result.

- **Precompute GPT-4 per-sentence Pearson on the full 660K** (`scripts/phase1_precompute_gpt4_pearson.py`). Pure chunk arithmetic on already-tokenised text — ~5 min on a login node, no GPU. Outputs `results/gpt4_pearson_full.json` (index → Pearson, chunks, latency).
- **Stratified-sample 200 sentences** by fixed absolute Pearson bins (`monotone ≥ 0.90`, `mild 0.70–0.90`, `reordering < 0.70`), ~70 per bin. Outputs `results/phase1_gate1_indices.json`.
- **Submit two jobs in parallel:**
  - `phase1_tau_sweep_ot_n200.pbs` — Gemma-4-E2B base + raw + OT + extended tau grid `{0.30, 0.50, 0.70, 1.00}`, 200 sentences, 02:30:00 walltime.
  - `phase1_tau_sweep_js_n200.pbs` — same but JS criterion, 00:30:00 walltime. Cheap ablation.
- **Run `scripts/phase1_reordering_bin.py`** on both `matrices.jsonl` outputs. Reports per bin: mean chunk-count delta vs GPT-4, per-sentence Pearson (ours at matched-count tau), MATCH rate under threshold 0.85.

**Pass criteria (see `TIMELINE.md` Gate 1):**
- Monotone bin: tie GPT-4 on chunk-count delta and Pearson.
- Reordering bin: strictly higher MATCH rate than degenerate baseline.
- METHOD §8 sanity checks all green on the winning tau.

**Unlocks:** Phase 2 SFT (10K annotation → matched-condition training → WMT15 newstest2015 extrinsic eval).

## 2. Cross-backbone sanity: Qwen3.5-2B (H6)

Same recipe as winning config, tests family-robustness. Do this after Gate 1 lands — need the Gemma anchor.

- Check on-disk Qwen3.5-2B: base or -it? If -it, fetch base via copyq.
- Repeat the winning config (base + raw + OT, extended tau) on the same 200 stratified indices.
- Compare per-bin stats to Gemma.

**Unlocks:** "cross-family robust" claim in the paper.

## 3. Scale-up to gemma-4-E4B (only if Gate 1 passes on E2B)

Tests H7. Gated per HOUSEKEEPING §1 SU-spend rule.

- Download `google/gemma-4-E4B` — ~10 GB copyq job.
- Repeat winning config on same 200 stratified indices; compare per-bin stats.
- If E4B produces higher per-bin performance than E2B, mention as scale-consistency evidence in the paper; don't over-claim.

**Unlocks:** scale ablation in the paper's Table.

## 4. Onwards to Phase 2 (SFT)

Only after Phase 1 conclusion is defensible.

- Annotate 10K then 50K sentences with the winning criterion (matches EAST Fig. 6's data-size trajectory).
- Build the SFT wrapper (trl.SFTTrainer per HOUSEKEEPING §4, not LLaMA-Factory).
- Fine-tune both conditions A (GPT-4 tags) and B (ours) on the same base backbone.
- Extrinsic eval on WMT15 newstest2015: BLEU/COMET/BLEURT vs AL/LAAL/**AL-CA**.
- **Gate 2:** an SFT run completes and emits tags in sensible places.
- **Gate 3:** the primary comparison exists.

Timeline weeks 6–10; see `../TIMELINE.md` Phase 2.

## Blockers and non-blockers, right now

**Blockers on the primary result:**
- Phase-2 SFT is downstream of Gate 1. Ordering matters.

**Blockers on Phase 3 (RWTH appendix) but not on Gate 1 or Phase 2:**
- Choice of RWTH baseline (compare our tags against what, since GPT-4 chunks are not available for RWTH's sentences)? Options: fast_align commits, monotonic wait-k floor, GPT-4 API re-annotation of the 509. Recommend GPT-4 API re-annotation (most direct comparison, ~$5-20 API cost). Decide before writing `src/eval/rwth_intrinsic.py`.

**Not blockers (deferrable):**
- Off-Multi-120K assembly (only Stretch A).
- Stage-II LoRA (only after Gate 3).
- BLEURT-20 fetch (only when we get to Phase 2 metrics).
- Doc-level and conversational SiMT (Stretches B, C).

## Weekly checkpoint reminder

Bring `LOG.md` to Dipankar meetings, not a summary — HOUSEKEEPING §1.
