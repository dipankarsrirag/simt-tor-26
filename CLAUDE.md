# Teacher-Free Read/Write Annotation for Simultaneous Machine Translation

Undergraduate research project. Supervisor: Dipankar Srirag (UNSW). Target venue: ACL/EMNLP Findings or IWSLT.

## The claim

[EAST](https://aclanthology.org/2025.findings-acl.1045.pdf) (Findings of ACL 2025) teaches an LLM adaptive read/write behaviour by fine-tuning on data where **GPT-4** decided where the read/write tags go. We replace GPT-4 with the backbone model's own predictive distributions: for each parallel pair, we hold the full source at data-construction time, measure when each target token's next-token distribution has converged to its full-source value, and place `<|eor|>` there.

**Falsifiable claim:** backbone-derived tag placement matches or beats GPT-4-derived placement, with the margin growing on word-order-divergent pairs.

**Why it should work:** EAST's own Appendix C discards training examples with unequal source/target chunk counts, which they note "often result from non-monotonic translations." Their policy is learned from a corpus with the reordering cases deleted. Ours needs no such filter.

These are constructed illustrations of the mechanism — I don't have the dataset loaded. The measurement at the bottom is what turns the claim into evidence, and you should do it before the claim goes in a paper.

**Separable prefix verb.** `Die Kommission kündigte gestern neue Maßnahmen zur Bekämpfung der Inflation an.` → "The Commission announced new measures to combat inflation yesterday."

`ankündigen` splits across the whole sentence. Read `Die Kommission kündigte` alone and the verb means *terminated*. The correct reading is only fixed by `an`, the final token. GPT-4's options under a monotonic instruction: one giant chunk (correct, latency ≈ full sentence), commit the wrong verb (caught by BLEURT < 80), or defer the verb to a later target chunk (chunk contents stop aligning 1:1).

Ours: `i*` for the English token *announced* doesn't fire until `i = n`. The tag lands exactly there. Commit points are per-token and are allowed to be late — there's no chunk-count constraint to violate.

**Fronted object.** `Diesen Vorschlag hat das Parlament im Juni abgelehnt.` → "Parliament rejected this proposal in June."

Source is OBJ–AUX–SUBJ–TIME–PART; target is SUBJ–VERB–OBJ–TIME. The first source chunk has no monotonic English counterpart — starting with "This proposal" forces passive, which changes voice. Empty target chunk (count mismatch), voice change (BLEURT risk), or merge everything (high latency).

Ours gives *Parliament* an early commit point (once `Parlament` is read) and *rejected* a commit point at the final token. That's the correct non-monotonic dependency, expressed without needing anything to align.

**Verb-final subordinate clause.** `Er sagte, dass die Regierung den Vorschlag im letzten Moment zurückgezogen habe.`

This one probably *survives* the filter — GPT-4 merges the whole subordinate clause into one chunk and counts stay equal. But note what happened: the low-latency variant is unattainable, so this sentence contributes coarse chunks even when labelled `low`.

**That's the stronger version of your claim, and I'd lead with it.** It's not only that reordering cases get deleted — surviving ones get systematically coarser segmentation, so the *low-latency* training signal is disproportionately drawn from monotonic sentences. That's a bias in the data, not just a gap, and it predicts EAST underperforms specifically at low latency on divergent pairs.

**Make it a figure.** Sample WMT15 De-En at ≥20 words, run EAST's Figure 19 prompt, and record (a) unequal chunk counts, (b) BLEURT < 80, (c) mean chunk length. Stratify by a reordering statistic — Kendall's tau on the alignment permutation, or crossing-alignment fraction from awesome-align. If drop rate and chunk coarseness both rise with reordering, that's your Figure 1 and the paper's motivation stops being an assertion.

Half a day's work, no training, and it's a good Phase 0 task for the student — it forces them to actually understand the data construction before they touch it.

## Non-negotiable invariants

1. **The test set is never annotated.** Annotation is training-data construction only. At inference the model emits tags autoregressively from raw streaming source. No references, no full-source access, no oracle at test time.
2. **No reference or full-source signal reaches inference.** If a component needs the full source at test time, it is the wrong component.
3. **Matched comparisons only.** Our-tags vs GPT-4-tags must use the same sentences, same count, same backbone, same hyperparameters. Anything else is not evidence.
4. **Report AL-CA, not just AL.** Computation-aware latency is where policy methods die. Ours should stay near-offline because all expensive computation is offline.

## Repository docs

| File | What's in it |
|---|---|
| `METHOD.md` | The annotation algorithm, precisely. Read before writing any code. |
| `EXPERIMENTS.md` | Ablation grid, baselines, metrics, evaluation protocol. |
| `TIMELINE.md` | Phases, milestones, and the gates that stop wasted work. |
| `RELATED.md` | What exists, what we build on, what not to re-derive. |
| `LOG.md` | Running log: decisions made and why; runs and outcomes. Append, never rewrite. |
| `HOUSEKEEPING.md` | Compute, paths, accounts, admin. Maintained by Dipankar. |

## Conventions

- Data: `SiMT-De-En-660K` (HuggingFace `biaofu-xmu/SiMT-De-En-660K`). This ships GPT-4 tags — that is the baseline condition, do not discard it.
- Backbone: same model for annotation and fine-tuning (see `METHOD.md` §5 for why, and the ablation that tests it).
- Every experiment gets a `LOG.md` entry with config, command, and result before the next one starts.
- Any deviation from `METHOD.md` or `EXPERIMENTS.md` gets logged as a decision, not made silently.

## Where to start

Read `METHOD.md`, then `TIMELINE.md` Phase 0. Do not start writing the training pipeline — the annotator is the project, the SFT is plumbing.