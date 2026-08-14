# Experiments

## Primary result

**Matched comparison.** Same sentences, same count, same backbone, same hyperparameters. Only the tag source differs.

| Condition | Tags from |
|---|---|
| A (baseline) | GPT-4 — as released in `SiMT-De-En-660K` and `SiMT-Multi-90K`|
| B (ours) | Backbone, per `METHOD.md` |

Report BLEU/COMET against AL, LAAL, and AL-CA. This is the headline; everything else supports it.

## Two evaluations, not one

**Extrinsic** — translation quality vs latency, as above. What reviewers look at first.

**Intrinsic** — annotation quality directly, no routing through translation. EAST's Appendix E.4 supplies the tool: the RWTH De→En manually aligned corpus, scoring the proportion of gold-aligned source tokens read before each target token is generated (their Eq. 4). Human alignments, independent of both annotators.

Score condition A's tags and condition B's tags on the same measure. This isolates annotation quality from everything downstream and is the cleanest evidence we have. Do not skip it because the extrinsic result looks good.

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

**Test set:** WMT15 De→En, matching EAST's primary setup. Extend to Ar-En and En-Zh only after De-En is complete — those are where the word-order claim lives, but they are extension, not entry.

## Guardrails

- Nothing is ever computed on test sentences. Not the criterion, not the tags, not a threshold.
- `tau` is selected on dev, never on test.
- No reference access anywhere in the inference path.
- Every run logged in `LOG.md` before the next begins.