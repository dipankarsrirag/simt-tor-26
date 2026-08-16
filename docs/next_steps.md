# Next steps, in order

Ordered by priority. Each item states what it does, why now, and what unlocks after.

## 1. RWTH intrinsic evaluation (unblocked)

RWTH data landed at `data/rwth-de-en/DeEn/`. All we need is the eval script. **This is the Gate-1 arbiter and the primary Phase-1 result.**

- Write `src/eval/rwth_intrinsic.py`.
  - Loads the 509 De/En sentences (handle Latin-1 encoding for `de`).
  - Parses `alignmentDeEn.talp` into per-sentence `(src_word_pos, tgt_word_pos)` alignment lists.
  - Runs the annotator on each De/En pair producing a per-target-*token* commit trace.
  - Maps word-level gold alignment → token-level `a_i` (per target token, the max source *token* position it aligns to, following word→token bijection under `GemmaTokenizer`).
  - Computes `A = (1/T) Σ I[a_i ≤ g_i]` per sentence and averaged over the 509.
- Also compute the same `A` for **GPT-4's tags** on the same 509 — this needs re-annotating those 509 with GPT-4's segmentation, or an alternative: use GPT-4's chunk-based commit trace derived from `source_chunks`/`target_chunks` if we can obtain GPT-4 chunks for these RWTH sentences (we cannot — RWTH is a separate corpus from EAST's WMT-derived training set). Alternative: use another automated baseline like fast-align commits at chunk granularity, and compare our tags against fast-align derived commits. Log the decision.
- Report: mean A, distribution over 509, per-sentence differences between ours (base+raw+JS at τ=0.10 and τ=0.15) and the chosen baseline.

**Unlocks:** the primary Phase-1 conclusion. If ours beats the baseline on A, the project's central claim holds at 2B scale. If not, iterate on criterion / prompt / backbone before scaling.

## 2. Wait for and interpret the OT sweep (already running)

Job `176307323`. When it lands (~1 hour walltime):

- Same offline analyses as Config C: `phase1_random_floor.py`, `phase1_entropy_sweep.py` (not applicable — matrices are OT, not entropy), `phase1_gpt4_pearson.py`, `phase1_per_sentence_compare.py`.
- Compare OT vs JS at matched chunk count: does OT dislodge the diagonal-tracking? Does per-sentence r(GPT-4, ours) rise above 0.175?
- Walk the 8 reordering candidates under OT (do more of them MATCH than under JS?).
- Update `experiments.md` with Config D's full table.

**Unlocks:** verdict on H5 (OT vs JS). If OT is materially better, it's the paper's headline. If OT ≈ JS, drop OT framing, ship JS with a shorter method section — still a paper.

## 3. Extend to ~200 sentences on the winning config (Config C or D)

Current n = 48 gives wide CIs on per-sentence r. Cost: ~5 min GPU for JS, ~15 min for OT.

- Adjust `--n_sentences 201 --max_src_tokens 80` in `phase1_tau_sweep.py`; submit.
- Re-run all offline analyses.

**Unlocks:** aggregate stats defensible for a paper. r-of-Pearsons with n=200 has meaningful CI.

## 4. Cross-backbone sanity: Qwen3.5-2B (base if we can find one)

Tests H6. If Qwen shows the same qualitative behaviour on the same 48 sentences (JS beats random at some tau; catches reordering candidates; RWTH A comparable to Gemma), the finding is family-robust.

- Check if `Qwen3.5-2B` on disk is base or -it. On HF Qwen usually publishes both; the on-disk name doesn't disambiguate. If -it, may need to fetch base — 4 GB copyq job.
- Repeat Config C recipe. Compare against Gemma-4-E2B on the same 48 (matrix per-sentence Pearsons, per-sentence r vs GPT-4).

**Unlocks:** "cross-family robust" claim in the paper.

## 5. Scale-up to gemma-4-E4B (only if Gate 1 passes on E2B)

Tests H7. Gated per HOUSEKEEPING §1 SU-spend rule.

- Download `google/gemma-4-E4B` — ~10 GB copyq job.
- Repeat winning config on same 48 (compare) and then 200 sentences.
- If E4B produces higher RWTH A than E2B, mention as scale-consistency evidence in the paper; don't over-claim.

**Unlocks:** scale ablation in the paper's Table.

## 6. Onwards to Phase 2 (SFT)

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

**Blockers on next work but not the primary result:**
- Choice of RWTH baseline (compare our tags against what, since GPT-4 chunks are not available for RWTH's sentences)? Options: fast-align commits, monotonic-wait-k floor, an independent LLM annotator like GPT-4 API. Decide before writing `src/eval/rwth_intrinsic.py`.

**Not blockers (deferrable):**
- Off-Multi-120K assembly (only Stretch A).
- Stage-II LoRA (only after Gate 3).
- BLEURT-20 fetch (only when we get to Phase 2 metrics).
- Doc-level and conversational SiMT (Stretches B, C).

## Weekly checkpoint reminder

Bring `LOG.md` to Dipankar meetings, not a summary — HOUSEKEEPING §1.
