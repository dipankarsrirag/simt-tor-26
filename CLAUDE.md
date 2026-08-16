# Teacher-Free Read/Write Annotation for Simultaneous Machine Translation

Undergraduate research project. Supervisor: Dipankar Srirag (UNSW). Target venue: ACL/EMNLP Findings or IWSLT.

## The claim

[EAST](https://aclanthology.org/2025.findings-acl.1045.pdf) (Findings of ACL 2025) teaches an LLM adaptive read/write behaviour by fine-tuning on data where **GPT-4** decided where the read/write tags go. We replace GPT-4 with the backbone model's own predictive distributions: for each parallel pair, we hold the full source at data-construction time, measure when each target token's next-token distribution has converged to its full-source value, and place `<|end-of-read|>` there.

**Falsifiable claim:** backbone-derived tag placement matches or beats GPT-4-derived placement, with the margin growing on word-order-divergent pairs.

**Why it should work:** EAST's own Appendix C discards training examples with unequal source/target chunk counts, noting these "often result from non-monotonic translations." Their policy is learned from a corpus with the reordering cases deleted. Ours needs no such filter — commit points are per-token and are allowed to be late.

**Motivating construction (fuller argument in the original CLAUDE.md thread).** German verb-final and fronted-object cases (e.g., separable-prefix `ankündigen ... an`) force EAST into three bad options: one giant chunk (right, but high latency), commit the wrong verb (semantic error), or defer to a later chunk (chunk counts mismatch and the row is dropped). Ours lets *announced* commit at the final source token without any chunk-count constraint to violate.

**Empirical status.** See `docs/experiments.md` for the current Phase-1 findings. Short version as of Aug 2026: end-to-end pipeline works, base-model + raw-concat + JS matches GPT-4 chunk counts and catches the walked reordering case (idx=553850) that -it + chat missed; OT sweep in flight; RWTH intrinsic eval (Gate 1 arbiter) unblocked.

## Non-negotiable invariants

1. **The test set is never annotated.** Annotation is training-data construction only. At inference the model emits tags autoregressively from raw streaming source. No references, no full-source access, no oracle at test time.
2. **No reference or full-source signal reaches inference.** If a component needs the full source at test time, it is the wrong component.
3. **Matched comparisons only.** Ours-tags vs GPT-4-tags must use the same sentences, same count, same backbone, same hyperparameters. Anything else is not evidence.
4. **Report AL-CA, not just AL.** Computation-aware latency is where policy methods die. Ours should stay near-offline because all expensive computation is offline.

## Where to start reading

**Read `docs/README.md` first — it is the index for the entire project.** In particular:

- `docs/method_overview.md` — how the annotator works, mechanically.
- `docs/hypotheses.md` — the falsifiable hypotheses (H1–H7) that motivate each experiment.
- `docs/experiments.md` — the Phase-1 runs and results, cross-referenced to hypotheses.
- `docs/data.md` — datasets, formats, RWTH access instructions, Gate-1 metric.
- `docs/next_steps.md` — what to do next, in order, with blockers.

The single-source-of-authority files at the repo root are:

| File | What's in it |
|---|---|
| `METHOD.md` | The annotation algorithm, precisely. |
| `EXPERIMENTS.md` | Ablation grid, baselines, metrics. |
| `TIMELINE.md` | Phases, milestones, gates. |
| `RELATEDWORKS.md` | Prior work we build on. |
| `LOG.md` | Append-only run + decision log. **The primary record.** `docs/` is a curated summary; `LOG.md` is what actually happened. |
| `HOUSEKEEPING.md` | Compute, paths, accounts, PBS conventions, venv discipline. |
| `OPTIONALS.md` | Paper strategy: venue verdict, blockers, method-improvement backlog, positioning. |

## What EAST actually does with its data (kept here — dataset-level context)

EAST is a **two-stage** recipe over **three datasets**. Ours mirrors the shape but scopes to Stage I.

| Dataset | HF | Size | Role in EAST | Role in our project |
|---|---|---|---|---|
| `SiMT-De-En-660K` | `biaofu-xmu/SiMT-De-En-660K` | 660,876 rows (De→En only) | **Stage I** — full-weight SFT, 1 epoch. Activates adaptive read/write. Derived from WMT15 De→En training. | **Primary run.** Both conditions (A = GPT-4 chunks, B = ours) trained on this. |
| `SiMT-Multi-90K` | `biaofu-xmu/SiMT-Multi-90K` | 90.7K rows across 8 directions | **Stage II** — LoRA on top of Stage I. Generalises to multilingual. | **Stretch** — only after Gate 3 passes. |
| `Off-Multi-120K` | not on HF (assemble à la ALMA from WMT17-21) | 120K, 8 directions | **Stage II** — LoRA co-training to preserve full-sentence quality. | Only if Stage II runs. |

Chunks ship as `source_chunks`/`target_chunks` lists — EAST wraps them at load time with `<|end-of-read|>`/`<|end-of-write|>` and a latency indicator token (`low`/`medium`/`high`, ≈1/3 of the 660K each). Loss is computed on **source + target + special tokens**.

### Test sets

- **WMT15 De→En newstest2015** — primary SiMT evaluation, matches EAST Fig. 3.
- **WMT22 X↔En** (8 directions) — multilingual stretch (EAST Fig. 4) + doc-level (EAST §4.3, De/Ru→En).
- **RWTH De→En manual alignments** — EAST App. E.4 intrinsic annotation-quality measure (Gate 1). See `docs/data.md`.

## Conventions

- Backbone: same model for annotation and fine-tuning (METHOD §5). Currently `gemma-4-E2B` (base pretrained; **not** the -it variant — the switch is logged in `LOG.md` 2026-08-15, motivation in `docs/hypotheses.md` H3).
- The GPT-4 chunks shipped in `SiMT-De-En-660K` are the **baseline condition**; do not discard them.
- Every experiment gets a `LOG.md` entry with config, command, and result before the next one starts.
- Any deviation from `METHOD.md` or `EXPERIMENTS.md` gets logged as a decision, not made silently.
