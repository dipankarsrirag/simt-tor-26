# Hypotheses driving the experiments

The project's central falsifiable claim is in `../CLAUDE.md`. Four paper-facing hypotheses (P1-P4) structure the Results section; each has **claim / prediction / test / status**. Consolidated 2026-08-18 late from an earlier working set of 23 exploratory hypotheses (H1-H23) that guided individual experiments; the H* labels are gone and replaced by the four P* ones. Git history preserves the archaeology.

**Gate status updated 2026-08-22.** The Aug 21 v6 pivot and Aug 22 v6b work reintroduced a matched-backbone cond-A (Multi-90K GPT-4 chunks) baseline, replacing the "past-work verbatim" strategy from Aug 18. Gate A/B are being deprecated in favor of the new head-to-head:

- **Gate A' (new, live 2026-08-22)** = P1's headline on the matched-backbone comparison: our v6b-ctrl-merged3 (OT chunks + EAST §3.1 merge) matches or beats cond-A (GPT-4 chunks) on the 4 overlapping directions (de-en, en-de, ru-en, en-ru), and dominates on ar/vi where cond-A can't reach. **Status:** merged3 beats cond-A on de-en at low_medium latency (31.88 vs 30.90), ties on en-de, still losing by 1-2 BLEU on ru-en/en-ru. Full-scale N=1012 test pending.
- Gate A (Qwen family-robustness) and Gate B (vs Simul-LLM published) from the Aug 18 plan are **paused** — the paper story shifted from single-arm-vs-published to matched-cond-A-head-to-head. If reviewers press on family-robustness, we may re-fire Qwen on the v6b recipe as a rebuttal experiment.

---

## P1 — Chunk-placement quality drives streaming translation quality (headline)

**Claim.** Given identical backbone, identical SFT recipe, identical training sentences, and identical EAST framework (special tokens + interleave), replacing procedural wait-k chunk boundaries with backbone-derived per-token OT-convergence chunk boundaries produces materially higher streaming BLEU at matched Average Lagging.

**Prediction.**
- (i) Under wait-k inference (k∈{3,5,7}), OT-SFT beats WaitK-SFT by ≥ +2 BLEU averaged over the wait-k grid at matched AL. **This is Gate B.**
- (ii) Offline BLEU is unchanged between the two arms (null result — OT chunking doesn't degrade full-source translation).
- (iii) Absolute BLEU on published test sets (WMT15 De→En newstest2015 + WMT22 De→En newstest2022) lands in the range of past LLM SiMT methods at matched-latency: within a few BLEU of EAST-Stage-I-8B at 4×/66× disadvantage, and competitive with Simul-LLM / TransLLaMa / SimulPL / ITST at matched published bins.

**Test.** OT-SFT (already trained, `sft_n10k/final/`). Streaming eval on newstest2013 dev, then WMT15 + WMT22 De→En for competitor comparison. Cross-paper plotting via `scripts/phase2_plot_bleu_al.py` — competitor numbers hand-populated from published tables per docs/related-work.md.

**Empirical status (as of 2026-08-22, N=50 FLORES devtest sanity):**
- (i) 🟡 PARTIAL — v6b-ctrl-merged3 (OT + EAST §3.1 merge) is +4.36 BLEU over raw v6b-ctrl on the matched-cond-A 4 directions, closing 76% of the +5.72 gap that separated raw OT from GPT-4 chunks. Still 1.36 BLEU behind cond-A on average. **On de-en at low_medium latency, merged3 (31.88) beats cond-A (30.90).**
- (ii) 🟢 Offline BLEU comparable across variants; the merged3 recipe doesn't degrade full-source translation.
- (iii) 🟡 PARTIAL — WMT15 De→En sanity (N=50) puts v6b-ctrl at BLEU 26.68 / AL 2.78 (low latency), 37.36 / AL 9.70 (high) → interpolated to ~35 BLEU at AL 8, matching EAST-Stage-I-Llama-2 (8B) with our 2B backbone. **Full N=1012 sweep on merged3 is the next milestone.**

**Historical note.** Prior to 2026-08-22, the paper's headline was OT-SFT vs Simul-LLM published wait-k=5 (Gate B). The Aug 21 v6 pivot (chat-template + NL latency prompt) + Aug 22 v6b work (direct-ids splice + α=1 + EAST §3.1 merge) reframed the story around matched-cond-A head-to-head. Earlier "vs cond-A n=10K" numbers (32.54 vs 32.41 offline; +5 BLEU under wait-k) in the archaeological section of `05-phase2_sft_and_streaming.md` are pre-v6-pivot and no longer represent the live claim.

---

## P2 — The finding is robust across backbone family, scale, and language pair

**Claim.** OT-SFT's absolute BLEU (P1's (iii)) and its lead over WaitK-SFT (P1's Gate B) reproduces on other backbones (Qwen3.5-2B, Gemma-4-E4B) and on multi-lingual training corpora (Multi-90K's four En↔X pairs) with a single fixed τ — showing the result is not a Gemma-2B/De→En artefact.

**Prediction.**
- (i) On Qwen3.5-2B at n=10K/De→En, OT-SFT lands within ~2 BLEU of the E2B result at matched wait-k, and beats past-work published wait-k=5 De→En numbers by ≥ +2 BLEU. **This is Gate A.**
- (ii) On Gemma-4-E4B (base, 4B params) at n=10K/De→En, OT-SFT absolute BLEU rises by 1-3 BLEU vs E2B (larger model); still beats published wait-k numbers by ≥ +2 BLEU.
- (iii) On Multi-90K's 4 pairs (En↔{De, Zh, Cs, Ru}) trained as a mixed 40K corpus with a single τ=0.30, OT-SFT lands in EAST Table 2 range averaged across pairs.
- (iv) τ=0.30 is within 1 BLEU of the per-pair optimal τ on all 4 directions (τ-generalisation).

**Test.** Qwen replication (annotation COMPLETE, dataset built, SFT queued); E4B replication (annotation COMPLETE, dataset build not yet triggered); Multi-90K mixed SFT (planned Weeks 5-6).

**Empirical status (as of 2026-08-22):**
- (i) Qwen deprioritized after v6 pivot; may re-fire as rebuttal experiment on the v6b recipe.
- (ii) E4B v6b-ctrl ✓ trained on E2B-annotated data (confound: 4B model learning to reproduce chunks a 2B model chose). Mean BLEU +3.21 over E2B ctrl on 40 cells. **BUT: merged3 (E2B, 2B) beats E4B (raw OT, 4B) by −0.49 BLEU** — chunk simplification outperforms scaling for this task. Clean scaling test (E4B annotator → E4B chunks → E4B SFT) needs the batched annotator (see LOG); deferred.
- (iii) Multilingual v6b covers 8 pairs (de-en, en-de, ar-en, en-ar, ru-en, en-ru, vi-en, en-vi) with a single fixed τ=0.30. **Extension to ar/vi is a coverage story cond-A cannot match** — SiMT-Multi-90K only ships de-en/en-de/ru-en/en-ru/zh-en/en-zh/cs-en/en-cs.
- (iv) τ=0.30 stayed after 2026-08-22 tau-sweep on de-en confirmed the annotator's chunk-per-sent regime (~6/sent at τ=0.30) IS the correct operating point; the "too-fine chunks" problem was at inference (α=5 issue), not annotation.

---

## P3 — OT-SFT is a policy-agnostic partial translator, not an adaptive commit policy (mechanism + limitation)

**Claim.** The mechanism behind P1's win is NOT that OT-SFT learned autonomous adaptive commit at n=10K/2B. It's that OT-derived chunks (with allowed reordering, variable length, no chunk-count filter) train the LLM to produce correct partial translations under *any imposed* streaming policy. Adaptivity is a separable capability that requires either (a) more training data, (b) loss reweighting on EAST special tokens, or (c) collapse-row filtering (already implemented in the 2026-08-18 dataset build via τ-fallback ladder); investigated as follow-ups.

**Prediction.**
- (i) Under check_argmax, OT-SFT gives chunks/sent = 1.00 (never voluntarily emits EOR mid-source) at n=10K on v1 dataset (collapse-heavy).
- (ii) OT-SFT's BLEU-vs-AL curve under wait-k lands above ITST / SM² published points at matched AL — OT-SFT is uniformly better across imposed commit positions than encoder-decoder-tradition SiMT at matched-latency.
- (iii) The class-imbalance mechanism: ~90% of SFT loss labels are content tokens, ~10% are EAST specials → insufficient gradient signal to induce autonomous EOR emission at 10K rows.
- (iv) At least one of the following induces chunks/sent > 1 under check_argmax without hurting wait-k BLEU: soft commit criterion at inference (Test A), loss upweighting on special tokens (Test B), retraining on the v2 dataset built with τ-fallback ladder (Test C — dataset ready as `sft_dataset_n10k_v2.json`).

**Test.** (i) confirmed empirically on v1. Follow-up: Tests A/B/C probe (iv).

**Empirical status:**
- (i) ✓ CONFIRMED at n=10K/E2B on v1 (collapse-heavy): chunks/sent = 1.00 under check_argmax.
- (ii) 🔄 PENDING: WMT15 De→En run for direct ITST/SM² comparison.
- (iii) ✓ Empirically demonstrated on real training rows: idx=2411 (collapse row) has 5.8% specials in loss; idx=372951 (positive row) has 14.5% specials. See `05-phase2_sft_and_streaming.md` walkthrough 2026-08-18.
- (iv) 🔄 PENDING: Tests A/B/C not yet run. Test C ready to submit (v2 dataset built 2026-08-18).

---

## P4 — The annotator is a universal preprocessing step (annotator-quality independence)

**Claim.** The chunk-annotation criterion (OT + τ=0.30) captures a property of the parallel data that is largely model-invariant — OT-SFT chunks from one backbone (E2B) can be used to SFT a different backbone (E4B, Qwen) with only mild degradation vs the matched-annotator setup. This makes the annotator a portable preprocessing step, not a per-backbone artifact.

**Prediction.**
- (i) E2B-annotated chunks → E4B-SFT is within 1-2 BLEU of E4B-annotated chunks → E4B-SFT at wait_k=5.
- (ii) Reverse (E4B-annotated chunks → E2B-SFT) shows slight lift or match.
- (iii) Cross-family (Gemma ↔ Qwen) is the harshest case: cross-family transfer degrades more than within-family (E2B ↔ E4B) but the transferred model still beats published wait-k numbers by ≥ +2 BLEU.

**Test.** Cross-annotator OT-SFT matrix — 6 off-diagonal cells (E2B/E4B/Qwen × 3 × OT-SFT). Planned Week 3-4.

**Empirical status:** 🔄 QUEUED. All 3 OT annotations complete (E2B ✓, E4B ✓, Qwen ✓); cross-family SFT + eval not started.

---

## Appendix — reported but not core hypotheses

- **RWTH-A intrinsic**: Phase 3 appendix; reviewer-expected, not a core claim.
- **Prompt-format ablation** (labelled `Source:...\nTranslation:...` raw-concat): method-improvement sub-check for the annotator; ablation table row.
- **Adaptivity investigations Tests A/B/C** (soft argmax at inference, EAST-token loss upweight, collapse-skip retrain): support P3's claim (iv); reported as discussion + one ablation table row each.


---

## Which experiment supports which prediction

| Experiment | Config | Reads out |
|---|---|---|
| OT-SFT training | Gemma-4-E2B base + 9,562-row v2 dataset (2026-08-18 fixes: fallback-τ ladder + latency reassignment) | Trains the primary model. |
| Streaming eval on newstest2013 | wait_k∈{3,5,7,check_argmax}; SacreBLEU-13a + AL + LAAL | P1 (i), P3 (i)-(ii). |
| Fig. 1 comparison — WMT15 De→En / AL | vs ITST, SM²/SimulMask, HMT, wait-k baseline (published verbatim); EAST dashed reference | P1 (iii) non-LLM tier. |
| Fig. 2 comparison — WMT22 De→En / LAAL | vs EAST, Simul-LLM, TransLLaMa, SimulPL, ConversationalSiMT (published verbatim) | P1 (iii) LLM tier; **Gate B** = OT-SFT ≥ +2 BLEU over Simul-LLM's published number. |
| Multi-90K mixed (weeks 5-6) | 40K rows across en↔{de,zh,cs,ru}, single τ=0.30 | P2 (iii)-(iv). |
| Qwen3.5-2B replication | annotation DONE; SFT queued | P2 (i), **Gate A**. |
| Gemma-4-E4B replication | annotation DONE; dataset build queued | P2 (ii). |
| Tests A/B/C (adaptivity) | soft argmax + special-token loss weight + collapse-skip retrain | P3 (iv). |
| Cross-annotator SFT matrix | 6 off-diagonal cells (E2B/E4B/Qwen × 3) | P4. |
| RWTH-A intrinsic (Phase 3 appendix) | 509 sentences with gold alignments; A-score vs GPT-4 / fast_align | Appendix — not a core P-claim. |
