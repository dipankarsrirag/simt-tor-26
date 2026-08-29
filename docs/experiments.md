# Experiments

## Scope: Stage I is the paper

EAST is a two-stage recipe (see `CLAUDE.md` §What EAST actually does): full-weight SFT on `SiMT-De-En-660K` (De→En, "Stage I"), then LoRA on `SiMT-Multi-90K` + `Off-Multi-120K` for multilingual generalisation ("Stage II"). **Our primary result is Stage I only.**

Why: (1) the claim lives in the annotation criterion, and Stage I is where that criterion decides tag placement — Stage II inherits Stage I's tags and just adds LoRA. (2) EAST reports Stage I separately (EAST-Stage-I in their Figure 3), giving us a published matched-comparison target. (3) A 14-week student project on a 2B backbone (`docs/setup.md` §5) has runway for Stage I plus ablations, not for a full Stage II sweep.

Stage II is a stretch — see `docs/_archive/TIMELINE.md`. Do not spend Phase 2 budget on it.

## Primary result

**Matched comparison, Stage I.** Same sentences (`SiMT-De-En-660K`, De→En), same count, same backbone, same hyperparameters. Only the tag source differs.

| Condition | Tags from |
|---|---|
| A (baseline) | GPT-4 chunks — the `source_chunks`/`target_chunks` fields shipped with `SiMT-De-En-660K`, wrapped with `<|eor|>`/`<|eow|>` and the `low`/`medium`/`high` latency indicator per EAST §3.2 |
| B (ours) | Backbone, per `docs/_archive/method-formal.md`. Same wrapping and indicators. |

Both conditions use EAST's loss recipe: cross-entropy on **source + target + special tokens** — not the target-only masking used by Wang et al. 2024. One epoch, full-weight tuning, matching EAST-Stage-I.

Report BLEU/COMET/BLEURT against AL, LAAL, and AL-CA on WMT15 De→En newstest2015. This is the headline; everything else supports it.

## Two evaluations, not one

**Extrinsic** — translation quality vs latency, as above. What reviewers look at first. **This is the paper's headline** (matches EAST Fig. 3's positioning).

**Intrinsic — Gate 1 (during Phase 1).** Stratified-by-reordering aggregate on 200 SiMT-660K sentences, using GPT-4's own per-sentence Pearson as the reordering-severity proxy (bins: ≥0.90 monotone, 0.70–0.90 mild, <0.70 reordering). Measures whether our tags track GPT-4 on monotone majority and catch the reordering minority. This is a **greenlight for Phase 2**, not a paper result — without gold alignment it measures agreement-with-GPT-4, not annotation quality.

**Intrinsic — Phase 3 appendix.** RWTH De→En manually aligned corpus (509 sentences, EAST App. E.4), scoring the proportion of gold-aligned source tokens read before each target token under Eq. 4. This is the human-aligned independent gold that condition-A and condition-B tags are both scored on. Lands in the paper's App. E, mirroring EAST's positioning.

Score condition A's tags and condition B's tags on both measures at their respective phases. Neither is optional. Do not skip Phase 3 RWTH because the extrinsic result looks good — a positive extrinsic without a positive intrinsic is a footgun at Findings review.

## Ablation grid

| Axis | Conditions | Question it answers |
|---|---|---|
| Divergence `D` | OT / KL / entropy-only / random-at-matched-latency | Does OT earn its cost? Does the full-source oracle do any work? |
| Annotator model | same-as-backbone / different model | Error amplification, or self-calibration benefit? |
| Monotonicity filter | with / without | Coverage gain on the non-monotonic examples EAST discards |
| Top-k support | 32 / 128 / 512 | Sensitivity; is the OT tractable at a useful `k`? |
| Data size | 10K / 50K / (100K) | Replicates EAST Fig. 6 and justifies our subset |

**Run order.** Divergence first (it decides the paper's framing), then annotator model, then monotonicity, then the rest. Do not run the full grid before the primary result exists.

## Design the monotonicity ablation carefully

Two distinct effects, easily confounded:

1. **Annotation quality** — our tags vs GPT-4 tags *on the sentences EAST kept*.
2. **Coverage** — our tags on the full set *including* the non-monotonic examples EAST discarded.

Run these as separate conditions. Reporting them together makes the result unreadable and invites a reviewer to assume the gain is all from extra data.

## Baselines

- EAST — cite published numbers for context (their Figure 3 and Tables 9–10 give the numerics).
- EAST re-run at our data size — needed for the matched comparison; cannot be cited.
- wait-k on the same backbone — the fixed-policy floor.
- Traditional SiMT (ITST, Mono-KD, SM²-Bi) and Conversational SimulMT — cite from EAST's Figure 3, do not reimplement.

Reproducing EAST's exact curve is a sanity check, not a deliverable. If it does not reproduce, the matched comparison is still valid as long as both conditions use our pipeline.

## Metrics

**Quality:** SacreBLEU; COMET (`wmt22-comet-da`); BLEURT-20. EAST showed BLEU and COMET disagree on En→Zh — report all three.

**Latency:** AL; LAAL; **AL-CA**; WWT (ms/word).

AL-CA is the one that matters for us. All our expensive computation is offline, so we should land near EAST's ~49 ms/word rather than the ~977 ms/word of prompt-updating wait-k. If we do not, something is wrong with the inference loop, not the method.

**Test sets.**
- **Primary — WMT15 De→En newstest2015.** Matches EAST Figure 3. This is the entry point; everything before the extension section runs here.
- **Stretch — WMT22 X↔En, 8 directions (De/Zh/Ru/Cs ↔ En).** Matches EAST Figure 4. Only touch after Gate 3 (`docs/_archive/TIMELINE.md`) passes and only if Stage II is being attempted — sentence-level SiMT numbers here require a Stage-II-trained model. Zh↔En and Cs↔En are where the reordering-divergence claim lives (fronted objects, verb-final subordinate clauses); those are the interesting directions if we run this at all.
- **Stretch — WMT22 De/Ru→En document-level.** EAST §4.3 zero-shot. Reuses the Figure-4 sentence data grouped by `docid`; no extra fetch. Only meaningful with a Stage-II model.

## Guardrails

- Nothing is ever computed on test sentences. Not the criterion, not the tags, not a threshold.
- `tau` is selected on dev, never on test.
- No reference access anywhere in the inference path.
- Every run logged in `LOG.md` before the next begins.