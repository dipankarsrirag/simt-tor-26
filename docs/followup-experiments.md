# Follow-up experiments — v6b → paper submission

Author: Dipankar Srirag · Draft: 2026-08-29 · **Status:** Decisions locked. Ready to draft training scripts.

Numbers throughout are targets, not results.

**EAST Backbone:** Llama-3-8B-Instruct.

---

## Research questions

Three questions, one per results-section story.

### RQ1. Recipe comparison (Figs 1, 2, 3)

Given the same backbone (EAST-8B), the same curated training corpus, and the same test sets, does self-annotation (backbone-derived OT chunks) beat every alternative training recipe — GPT-4 chunks (EAST as-shipped), machine-translated targets, wait-k, and Conversational SiMT (Wang et al. 2024)?

### RQ2. Backbone scaling (Fig 4)

Does the self-annotation → SFT recipe hold across backbone scale? Concretely: does moving from 2B (Gemma-4-E2B-it) → 4B (Gemma-4-E4B-it) → 8B (EAST-8B) monotonically improve simultaneous translation quality on the same curated-corpus with each backbone self-annotating?

### RQ3. Annotator portability (Fig 5)

Is the OT annotation a portable preprocessing step across backbones — i.e., can chunks derived from a small backbone (Gemma-2B) be reused to SFT any larger backbone (Gemma-4B, EAST-8B) without material loss, so annotation is a one-off cost rather than per-backbone?

---

## Resolved decisions

Answers to the three ambiguities from the previous draft (see git history for the raw options).

- **Q1 → Q1b.** `EAST↺east` = same curated sources with EAST-style (machine-translated) targets. Operational definition: sub-sample SiMT-660K + SiMT-Multi-90K stratified to match the curated-corpus row counts per direction (drop ar/vi since EAST corpus lacks them → effective total ~118K rows). This uses EAST's actual training targets as the "machine-target" reference; no additional retranslation cost.
- **Q2 → Fig 1 keeps the existing WMT-15 plot lines AND adds two EAST-family lines** (`EAST↺ours`, `EAST↺east`). Total 6 lines. If too crowded at render time, drop `Gemma-2B CondA` (superseded by `Gemma-2B Ours` in the paper narrative).
- **Q3 → Q3a.** Fig 3 = 3 lines (`EAST↺ours`, `EAST↺waitk`, `EAST↺conv`). Gemma-2B lines *not* on Fig 3 (would muddle the fixed-backbone-recipe-comparison story). Coverage story lives in Fig 4/5.
- **Q3 hedge (user note).** The backbone-comparison story may need additional plots for ar/vi if Fig 3 is EAST-only. Parked — decide after Fig 3 lands.

---

## Naming

| Symbol | Meaning |
|---|---|
| **Backbone** | The LM used both as annotator (offline) and as the SFT starting point. |
| **Corpus** | The parallel-text pool used to derive training rows. Two options: `east-corpus` (SiMT-660K + SiMT-Multi-90K, GPT-4 chunks shipped) or `curated-corpus` (europarl + news-comm + TED2020, our human-target curation; see `docs/data.md`). |
| **Annotator** | The model whose distributions decide OT chunk boundaries. Usually the same as **Backbone** (self-annotation) but can differ (cross-annotation). |
| **Policy** | The SFT-time chunking recipe: `ot` (ours), `wait-k` (fixed), `conv-simt` (Wang 2024), `gpt4-chunks` (EAST-shipped). |

A run is fully specified by `(Backbone, Corpus, Annotator, Policy)`.

Compact figure tags:
- **EAST** — released checkpoint `biaofu-xmu/EAST-8B` = `(EAST-8B, east-corpus, GPT-4, gpt4-chunks)`.
- **EAST↺ours** — `(EAST-8B, curated-corpus, EAST-8B, ot)`.
- **EAST↺east** — `(EAST-8B, east-corpus[N-matched to curated per-direction], EAST-8B, ot)`. Isolates target quality (human vs machine-generated) with source-pool + annotator + policy held fixed.
- **EAST↺waitk** — `(EAST-8B, curated-corpus, —, wait-k[k=k_wk])`.
- **EAST↺conv** — `(EAST-8B, curated-corpus, —, conv-simt[k=k_cv])`.

The `↺` glyph reads "self-trained on"; policy suffix disambiguates.

---

## Corpus sizes (pre-registered)

**curated-corpus** (from `_archive/results/gemma_2b_curated/sft_dataset_multilingual_v6b_htgt_final.json`, checked 2026-08-29):

| Direction | Rows |
|---|---|
| en-de | 29,811 |
| de-en | 28,836 |
| en-ru | 29,934 |
| ru-en | 29,412 |
| en-ar | 30,000 |
| ar-en | 30,000 |
| en-vi | 30,000 |
| vi-en | 30,000 |
| **Total** | **237,993** |

**Proportion-matched target for `EAST↺east` (Q1b):** stratified sub-sample of SiMT-660K + SiMT-Multi-90K to match the per-direction row counts above. Ar/vi rows are absent from EAST's corpus, so the effective total = 29,811 + 28,836 + 29,934 + 29,412 = **117,993 rows** across en↔de and en↔ru only. This asymmetry is unavoidable and must be flagged in the paper: `EAST↺east` cannot appear on Fig 3 (ar/vi) — see §Figures.

---

## Pre-registered hyperparameters

**Wait-k baseline (`EAST↺waitk`).** Fix one *k* per latency bin, no per-bin tuning:

| Latency | k (wait-k source-reads before first write) |
|---|---|
| low | 3 |
| medium | 5 |
| high | 7 |

Rationale: matches EAST §4.2 wait-k grid; single value per bin prevents post-hoc "best-k" hyperparameter search that would unfairly inflate the wait-k line.

**Conv-SiMT baseline (`EAST↺conv`).** Wang 2024 (arXiv 2402.10552, §2):
alignment-derived READ/WRITE pairs `(Rⱼ, Wⱼ)` on a monotonic dependency
graph — NOT a fixed-k word window (`k_cv=4` in earlier drafts of this
doc was a mis-read of the paper; the algorithm has no k at training
time). Wang's original alignment tool is `fastalign` (Dyer 2013); we
substitute `awesome-align` (mBERT-based, pip-installable, higher-
precision) — deviation logged 2026-08-31 in `LOG.md`. Inference: Wang
reads `n ∈ {3,5,7,9,11,13}` tokens/step + RALCP for WRITE termination;
we simplify to greedy-per-chunk (read `n`, generate until `<|end-of-
write|>`) so all three baselines (OT / wait-k / conv-simt) share the
same streaming inference policy. Per-latency training data (single-model
vs 3 latency-labelled variants) still to pick — see LOG.md.

---

## Figures

### Fig 1 — Headline: WMT15 De→En

**Question (RQ1 on the primary EAST test set).** Under matched training conditions on WMT15 De→En, does self-annotation improve on the EAST recipe?

**Test set.** WMT15 newstest2015 De→En, N=2169. 4 lines already landed; 2 new.

**Lines (single subplot, BLEU vs AL).**

| Line | Backbone | Corpus | Annot. | Policy | Status |
|---|---|---|---|---|---|
| EAST | EAST-8B | east-corpus | GPT-4 | gpt4-chunks | ✓ landed |
| Gemma-2B CondA | Gemma-4-E2B-it | multi-90k | GPT-4 | gpt4-chunks | ✓ landed |
| Gemma-2B CondB | Gemma-4-E2B-it | multi-90k | Gemma-4-E2B-it | ot | ✓ landed |
| Gemma-2B Ours | Gemma-4-E2B-it | curated-corpus | Gemma-4-E2B-it | ot | ✓ landed |
| **EAST↺ours (new)** | EAST-8B | curated-corpus | EAST-8B | ot | ✗ needs training + eval |
| **EAST↺east (new)** | EAST-8B | east-corpus[N-matched] | EAST-8B | ot | ✗ needs training + eval |

**Cost delta.** 2 new EAST-8B checkpoints × 5 latencies = 10 new eval cells on WMT15 alone.

**Render note.** 6 lines is at the crowding limit. If unreadable, drop `Gemma-2B CondA` (superseded by `Gemma-2B Ours`).

---

### Fig 2 — WMT22, high-resource: En↔{De, Ru} × X↔En

**Question (RQ1 on WMT22).** With the EAST backbone fixed, which axis — corpus, target text, or policy — drives the win?

**Test sets.** WMT22 newstest2022 de-en (N=1979), en-de (N=1904), ru-en (N=2016), en-ru (N=2037). All already landed for Gemma-2B and EAST-8B.

**Layout.** 4 subplots — (a) de-en, (b) en-de, (c) ru-en, (d) en-ru.

**Lines per subplot (all use EAST-8B backbone; 3-latency ladder to match EAST training).**

| Line | Corpus | Annot. | Policy | What it isolates |
|---|---|---|---|---|
| EAST | east-corpus (full) | GPT-4 | gpt4-chunks | published SoTA reference |
| EAST↺ours | curated-corpus | EAST-8B | ot | **full recipe** — new data + self-annotation + OT |
| EAST↺east | east-corpus[N-matched] | EAST-8B | ot | **target quality** — human curated targets vs machine EAST targets, holding annotator+policy fixed |
| EAST↺waitk | curated-corpus | — | wait-k (k per latency, pre-reg above) | **policy** — is the win the OT annotation or does any streaming training on curated help? |
| EAST↺conv | curated-corpus | — | conv-simt (k_cv=4) | **policy alternative** — vs current LLM-SiMT competitor |

**Conv-SiMT recipe (Wang et al., 2024).** Segment parallel sentences with `awesome-align` → format as multi-round dialogue → SFT. Inference reads *k_cv* tokens per step and incrementally decodes. Net-new module `src/train/conv_simt.py`.

---

### Fig 3 — IWSLT, low/mid-resource: En↔{Ar, Vi} × X↔En

**Question (RQ1 on unseen-by-EAST language pairs).** Does the recipe generalise to language pairs EAST's training data does not cover, using only Llama-3-8B pretraining coverage?

**Test sets.** IWSLT17 en-ar / ar-en (N=1460), IWSLT15 en-vi / vi-en (N=1268). Ar/vi cells already landed for Gemma-2B; nothing landed for EAST-8B (out of training scope — see Q5-parked below).

**Layout.** 4 subplots — (a) en-ar, (b) ar-en, (c) en-vi, (d) vi-en.

**Lines per subplot (all use EAST-8B backbone; 3 lines per Q3a decision).**

| Line | Corpus | Annot. | Policy |
|---|---|---|---|
| EAST↺ours | curated-corpus | EAST-8B | ot |
| EAST↺waitk | curated-corpus | — | wait-k |
| EAST↺conv | curated-corpus | — | conv-simt |

**Omitted intentionally.**
- No `EAST` — released weights untrained on ar/vi.
- No `EAST↺east` — EAST corpus has zero ar/vi rows (see §Corpus sizes).
- No `Gemma-2B Ours` per Q3a — keeps Fig 3 a pure EAST-backbone recipe comparison. The Gemma-2B vs EAST-8B comparison on ar/vi lives in Figs 4/5 instead.

---

### Fig 4 — Backbone scale × self-annotation (RQ2)

**Question (RQ2).** Does self-annotation help every backbone, or is it a 2B artifact?

**Test sets.** Same 8 directions as Figs 2+3.

**Layout.** 2×4 grid. Top row: En→{De, Ru, Ar, Vi}. Bottom row: {De, Ru, Ar, Vi}→En.

**Lines per subplot (3 backbones, each self-annotated on curated-corpus with OT policy).**

| Line | Backbone | Corpus | Annot. | Policy | Status |
|---|---|---|---|---|---|
| Gemma-2B Ours | Gemma-4-E2B-it | curated-corpus | Gemma-4-E2B-it | ot | ✓ landed |
| Gemma-4B Ours | Gemma-4-E4B-it | curated-corpus | Gemma-4-E4B-it | ot | ✗ annotate + train |
| EAST↺ours | EAST-8B | curated-corpus | EAST-8B | ot | ✗ annotate + train |

**Reads.**
- Monotone-up in backbone size → self-annotation is real, scales.
- Flat / inverted → three possible causes to disentangle:
  1. Self-annotation is a small-model artifact (the story shrinks).
  2. 8B is **data-bound at 30K rows/direction**, not scale-bound — larger backbones need more data before scale pays off. Fixable by adding a smaller-N and larger-N point per backbone; not fixable inside this figure.
  3. Annotation cost dominates any scale gains (relevant to Fig 5's story).

The paper should hedge cause (2) explicitly — otherwise a flat Fig 4 gets over-attributed to a method failure.

---

### Fig 5 — Annotator identity × backbone (cross-annotation; RQ3)

**Question (RQ3).** Do larger backbones need to self-annotate, or can they use a smaller model's OT plan?

**Test sets.** Same 8 directions.

**Layout.** 2×4, same as Fig 4.

**Lines per subplot (all annotated by Gemma-2B; SFT starting points differ).**

| Line | Backbone | Corpus | Annot. | Policy | Status |
|---|---|---|---|---|---|
| Gemma-2B (self) | Gemma-4-E2B-it | curated-corpus | Gemma-4-E2B-it | ot | ✓ landed (Fig 4 line 1) |
| Gemma-4B ← 2B annot | Gemma-4-E4B-it | curated-corpus | Gemma-4-E2B-it | ot | ✗ train (reuse 2B annotation) |
| EAST-8B ← 2B annot | EAST-8B | curated-corpus | Gemma-4-E2B-it | ot | ✗ train (reuse 2B annotation) |

**Reads.** Fig 5 tracking Fig 4 within ~1 BLEU per cell → **annotation is a portable preprocessing step**: run once with the smallest model, reuse tags for any downstream backbone. Big deployment story. Fig 5 significantly below Fig 4 → self-annotation is essential and per-backbone annotation cost cannot be amortised.

**Reverse direction (big-annotator → small SFT)** deferred to appendix table row, not a figure. See Q4-parked below.

---

## Runs required

Counting only runs not already landed.

### New annotations (offline, on curated-corpus)

| Annotator | Directions | Purpose | Cost (est.) |
|---|---|---|---|
| EAST-8B (Llama-3-8B) | 8 | Fig 1, 2, 3, 4 (EAST↺ours) | ~4× E2B annotation cost → **~40 GPU-h** |
| Gemma-4-E4B-it | 8 | Fig 4 middle line | ~2× E2B → **~20 GPU-h** |

Note: `EAST↺east` uses EAST's pre-shipped GPT-4 chunks on the sub-sampled Multi-90K/SiMT-660K rows — but we deliberately re-annotate those rows with EAST-8B OT to hold **annotator + policy** fixed across Fig 2 lines 2 & 3. So EAST-8B annotation must also cover the 118K sub-sampled east-corpus rows: +~6 GPU-h.

### New SFT runs

| Setup | Runs | Rationale |
|---|---|---|
| EAST↺ours (multilingual, 8 dirs) | 1 | Figs 1, 2, 3, 4 |
| EAST↺east (proportion-matched east-corpus) | 1 | Fig 1 line 6, Fig 2 line 3 |
| EAST↺waitk (multilingual) | 1 | Figs 2, 3 line: wait-k |
| EAST↺conv (multilingual) | 1 | Figs 2, 3 line: conv-simt |
| Gemma-4B self-ann → SFT | 1 | Fig 4 middle |
| Gemma-4B ← 2B annot SFT | 1 | Fig 5 middle |
| EAST-8B ← 2B annot SFT | 1 | Fig 5 right |
| **Total** | **7** | |

Each SFT run comparable to `v6b_v2bal_v3_htgt` — ~12 GPU-h on H200 (see `LOG.md` v6b_htgt training entries). SFT budget: **~85 GPU-h**.

### Extrinsic eval

- EAST-8B new checkpoints (4: ↺ours, ↺east, ↺waitk, ↺conv): 3 lats × varying dirs per figure.
  - ↺ours: WMT15 (1) + WMT22 (4) + IWSLT (4) = 9 dirs × 3 lats = **27 cells**
  - ↺east: WMT15 (1) + WMT22 (4) = 5 dirs × 3 lats = **15 cells** (ar/vi excluded — no training data)
  - ↺waitk: WMT22 (4) + IWSLT (4) = 8 dirs × 3 lats = **24 cells**
  - ↺conv: same as ↺waitk = **24 cells**
- Gemma-4B (2 checkpoints: self + cross): 5 lats × 8 dirs = 40 cells each → **80 cells**
- Total: **~170 new eval cells** at ~30-60 min each on H200 → **~130 GPU-h**.

### Grand total resource ask

- Annotation: **~66 GPU-h** (EAST-8B ×46 + Gemma-4B ×20)
- SFT: **~85 GPU-h**
- Eval: **~130 GPU-h**
- Redo/debug buffer: **~50 GPU-h**
- **~330 GPU-h total.** On ba39's ~50-job cap and H200 turnaround, fits in ~2 wall-clock weeks via `jobs/loop_resubmit.sh`.

---

## Metrics reported

Per cell:
- **SacreBLEU** (headline).
- **COMET** (`wmt22-comet-da`) for Figs 1-3 to catch BLEU-COMET disagreement (`docs/experiments.md` §Metrics).
- **AL** (Ma 2019 §4, word-unit).
- **AL-CA** (computation-aware) for Fig 1 only — the paper's promised streaming metric; verifies our offline-first design (~50 ms/word target).

Not yet computed at scale — see `docs/next-steps.md` §COMET rescoring.

---

## Parked ambiguities / low-priority

- **Q4 (reverse cross-annotation).** Big-annotator + small-SFT. Interesting but not a figure. Appendix table row: `Gemma-2B ← 8B annot` vs the Fig-5 pairs.
- **Q5 (EAST-8B on ar/vi).** EAST-8B pretraining was on de/zh/cs/ru; ar/vi rely on Llama-3-Instruct base coverage only. Default plan: train EAST↺{ours,waitk,conv} on the full 8-direction curated-corpus and let pretraining carry ar/vi. If Fig 3 shows EAST-8B collapses on ar/vi, either (a) note this as a hard limit of the recipe, or (b) drop ar/vi from Fig 3 and add a Gemma-vs-EAST comparison plot per Q3 hedge.
- **Q3 hedge (backbone-comparison on ar/vi).** If the reader wants to see how Gemma-2B compares to EAST-8B on ar/vi (per Fig 3's omitted Gemma line), that lives in Figs 4/5 subplots (c/d bottom row / g/h top row). If those don't tell the story clearly, add a dedicated 1×4 plot (ar-en, en-ar, vi-en, en-vi) comparing all three backbones at fixed self-annotation. Decide after Fig 4 lands.

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| EAST-8B self-annotation is prohibitively slow (8B forward passes per token) | High — blocks Figs 1-3 | Prototype on 100 sentences first; if >8× E2B, consider distilled annotator or subset |
| EAST↺ours matches or loses to EAST | Terminal for the headline | Report faithfully; pivot to "Gemma-2B Ours is efficient at 4× fewer params" story |
| Wait-k / Conv-SiMT reimplementation is buggy | Fig 2 line 4-5 unusable | Sanity-check against published wait-k numbers on WMT15 De→En; require ±1 BLEU match |
| curated-corpus target quality is uneven across directions | Cross-direction comparisons look noisy | Sample-inspect 20 rows/direction before large SFT runs |
| GPU quota exhaustion mid-run | Delays 1-2 weeks | Resubmit-loop already handles queue saturation (`jobs/loop_resubmit.sh`) |
| Fig 4 flat because 8B is data-bound at 30K rows/direction | Weakens scaling story | Note explicitly in analysis; add a smaller-N and larger-N point per backbone if the initial run is flat |
| Fig 1 with 6 lines is unreadable | Headline harder to parse | Drop `Gemma-2B CondA` (superseded by `Gemma-2B Ours`) — falls back to 5 lines |
| `EAST↺east` and `EAST↺ours` differ in both source and target simultaneously | Ablation isn't clean | Explicit in Fig 2 caption: "target quality controlled; source pool differs". The ideal orthogonal ablation (same sources, different targets) needs retranslation — deferred if reviewers ask. |

---

## What ships to the paper

- **Fig 1** → §5 headline. RQ1 on WMT15.
- **Fig 2** → §5.1. RQ1 recipe-axis decomposition on WMT22.
- **Fig 3** → §5.2. RQ1 out-of-training-scope generalisation on IWSLT.
- **Fig 4** → §5.3. RQ2 backbone scaling.
- **Fig 5** → §5.4. RQ3 annotator portability (deployment story).
- Metrics table (per direction, per latency, all 5 figures collapsed) → App. A.
- COMET / AL-CA comparison table → App. B.

Every figure result reported, positive or negative — selective reporting is a Findings-reviewer footgun (`docs/experiments.md` §Guardrails).
