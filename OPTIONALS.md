# OPTIONALS.md — improvements ranked by paper impact

The plan and method as written are **IWSLT-publishable** if the Stage-I primary lands. For **ACL/EMNLP Findings**, three blockers stand between us and a defensible submission — none of them fatal, all of them addressable within the 14-week window if scheduled now. **ICLR is the wrong venue** and should not be optimised for (representation-learning / algorithmic-breadth story, not what we have).

This document is ordered by paper impact, not by effort. Each item states the change, cites the closest prior work I read directly (not just the abstract), and says what the change adds over that prior work.

## Venue verdict

| Venue | Verdict | Precondition |
|---|---|---|
| **IWSLT** | Yes as-framed | Stage-I lands, RWTH intrinsic result is positive |
| **ACL/EMNLP Findings** | Plausible after §Blockers 1–3 | Scale framing fixed, REINA distinction sharp, exposure-bias measured |
| **ACL/EMNLP main track** | No | Would need 8B replication + monolingual-multilingual span |
| **ICLR** | No | Wrong shape — this is applied SiMT, not representation learning |

## Blockers for ACL/EMNLP Findings

### 1. Scale framing — preregister "at 2B" or add an 8B replication

**Problem.** EAST's headline numbers (Fu et al. 2025, Table 2) are on Llama-3-8B-Instruct. Ours are on Qwen3.5-2B (`HOUSEKEEPING.md` §5). Two soft-punt lines in the current docs — "matched comparison holds at any scale" (`CLAUDE.md`) and "matched comparison is still valid as long as both conditions use our pipeline" (`EXPERIMENTS.md` §Baselines) — will not survive a Findings review. Reviewers expect either (a) scale-matched evidence or (b) explicit scale-conditioned claim.

**Two clean paths, pick one:**

- **Option A — "At 2B" framing.** Rewrite the abstract, intro, and results section to say *"at the 2B scale we can afford"*. Add one paragraph in §Discussion explicitly limiting the claim's scope. All the ablations then live at 2B and require no additional compute. Cheapest and honest. If we run this, `LOG.md` gets a DECISION entry saying "declined scale replication; scoped to 2B" with the reasoning.
- **Option B — one 8B replication after Gate 3.** Rerun both conditions (A = GPT-4 chunks, B = ours) at `Llama-3.1-8B-Instruct` (on disk at `MODEL_BASE/Llama-3.1-8B-Instruct`) on the *same* WMT15 De→En test. This is ~4× the compute of a 2B run but on one job. Two H200s should suffice via `tensor_parallel_size=2`. This is the strongest version of the paper.

**Recommendation.** Option A during the 14 weeks. Add Option B as a post-writeup follow-up if time. Do not attempt B before Gate 3 passes — it burns SU on a story that doesn't yet exist.

### 2. REINA distinction — a full subsection, not a bullet

**Problem.** REINA (Hirschkind et al., AAAI 2026, arXiv 2508.04946) is structurally the closest published idea. `RELATEDWORKS.md` flags it in one sentence. Findings reviewers who read REINA and our paper back-to-back will not see the distinction unless we make it explicit at section level.

**What REINA actually does (§3.1 of the paper, page 3):**

Their criterion is the mutual-information gain of waiting for the rest of the audio:

```
F(a, S, n, t) := I(s_{n+1}; a_T, S_n) − I(s_{n+1}; a_t, S_n)
              = H(s_{n+1} | a_t, S_n) − H(s_{n+1} | a_T, S_n)
              = E[log p(s_{n+1} | a_T, S_n) − log p(s_{n+1} | a_t, S_n)]
```

That is: **the log-probability-of-the-next-token ratio under full vs partial input**, computed from the non-streaming translation model. They then train a policy head `q_θ` on top of the decoder to *predict* whether `F` is above a threshold, with monotonicity and L2 regularisation. At inference, the policy head runs per step; full audio access is training-only.

**Where we differ, in one paragraph:**

> REINA supervises a per-step *policy head* at training time with a full-vs-partial log-probability signal and queries that head during streaming inference. Our criterion uses the same underlying full-vs-partial signal, but we apply it **one stage earlier — during training-data construction**. Tag placement is decided offline once per sentence; at inference there is no policy head to query, no per-step supervised prediction, and no exposure-bias gap between the training-time oracle (reference-conditioned) and inference-time behaviour (self-conditioned). The KV-cache reuse that EAST inherits from its interleaved-format inference (~49 ms/word vs ~977 ms/word for prompt-updating wait-k, EAST §4.1) is preserved without modification.

**Additional rhetorical wins:**
- REINA's log-prob-ratio estimator is **exactly the KL-with-uniform-prior special case** of our distributional distance. If our OT-vs-KL ablation shows KL matches OT, we ship *REINA's signal applied to data construction* — a strictly cheaper and cleaner variant of a known-good criterion. If OT beats KL, our criterion is strictly stronger than what REINA uses.
- REINA needs their policy head to hit an accuracy ceiling to be useful (their §Inference Policy explicitly notes tuning difficulties around threshold α). Our decision quality caps at oracle quality, not policy-head quality.

**Effort:** ~2 pages of prose, one comparison diagram. Do this before submission, not during rebuttal.

### 3. Exposure bias — measure it on dev, do not just admit it

**Problem.** `METHOD.md` §9 says "measure if time allows; state regardless." That's not enough. The gap between reference-conditioned `P_pre[i][j] = p(y_j | S_≤i, T_<j)` (data-construction) and self-conditioned `p(y_j | S_≤i, ŷ_<j)` (inference) is directly quantifiable on dev — no ablation grid, just one forward pass at a handful of `tau` values.

**Concrete diagnostic to add to Phase 1** (after Gate 1, before Phase 2):

1. Sample 500 dev sentences.
2. For each, compute `i*[j]` under reference forcing (our data-construction protocol).
3. For each, run the trained model in inference mode, capture the model's *self-conditioned* commit points, call them `i*_hat[j]`.
4. Report `mean |i*[j] − i*_hat[j]|` and the fraction of tokens where the two disagree.

**Interpretation.** If the gap is small (say <10% of tokens diverge by more than 1 source token), state it and move on — this is a strength paragraph. If large, the paper reframes: "annotation quality is easier than policy learning; here is how much of the observed BLEU gain comes from that." Either result is publishable; the current "we admit it" line is not.

**Comparison to prior work.** DiG-SST (Chen et al., AAAI 2024, arXiv PDF §5 Inference Policy) explicitly cites this exposure bias as the reason they combine their adaptive policy with wait-k as a ceiling: *"the divergence-based policy model is trained with the reference translation as history, [so] there is inherent exposure bias during inference, which makes accurate prediction more challenging."* They mitigate it structurally by falling back to wait-k. **We sidestep it by moving the decision to data construction** — the exposure-bias failure mode affects one-shot annotation, not runtime, so the policy-network exposure-bias literature does not directly apply. Making that argument requires the measurement above; without it, it's a claim, not evidence.

**Existing exposure-bias literature to cite for context:**
- Bengio et al. 2015 (scheduled sampling) — the original diagnosis, RNN-era, teacher-forcing-vs-inference mismatch.
- Ranzato et al. 2016 (MIXER) — sequence-level REINFORCE as fix.
- Zhang et al. 2019 (Confidence-Aware Scheduled Sampling, arXiv 2107.10427) — LLM-era treatment, uses model confidence to gate teacher forcing. Their diagnostic can be adapted; their fix (scheduled sampling on the annotator) is orthogonal to us since our annotator runs once, offline.

## Strengthening (non-blocking, real paper-impact per hour)

### 4. Reordering-correlation Figure 1 — schedule as Phase 0, not "if time"

`CLAUDE.md` outlines this — Kendall's tau on the alignment permutation as the x-axis, plot drop rate and mean chunk length as the y-axis, stratified by reordering statistic. The doc estimates half a day. It's currently unscheduled.

**Why it's the paper's motivation figure.** Without it, the intro's core claim — *"EAST's data construction is biased against reordering examples"* — is an assertion. With it, it's evidence. Reviewers read Figure 1 first; if it lands the motivation, the rest of the paper coasts.

**Recommendation.** Move it from `CLAUDE.md`'s "half a day's work, no training" aside to `TIMELINE.md` Phase 0 as a concrete deliverable, alongside the data-format sanity check. It doesn't need the annotator built — it only needs `SiMT-De-En-660K` (already fetched) + awesome-align or fast_align for the alignment stat.

### 5. Multiple seeds + paired bootstrap

Standard at Findings; currently unspecified. The `LOG.md` template mentions `seed` but not multi-seed protocol.

**Concrete change:** every headline number in `RESULTS.md` (the table format is TBD) runs on **3 seeds** minimum (`42, 43, 44`). Report mean ± std. For the primary A-vs-B comparison, run **paired bootstrap on 1000 resamples of the test set** for BLEU/COMET/BLEURT differences; report p-value and 95% CI on the delta.

Add to `EXPERIMENTS.md` §Guardrails: *"Single-seed results are debugging output. Anything in a table gets three seeds and a paired bootstrap."*

Cost: 3× annotation and 3× SFT for the primary comparison. That is significant. Restrict multi-seed to the headline table + the divergence ablation; single-seed is fine for the top-k support / data-size sweeps.

### 6. Data-efficiency reframing — the "10K, no API cost" story

**Current framing** (implicit in `METHOD.md` §7): *"Annotate a subset (10K–50K) — EAST Fig. 6 shows most benefit at 10K."*

**Better framing for the paper:** *"Zero API cost. 10K sentences. Matched or better than EAST's 10K GPT-4-annotated."*

This is the same experiment, framed differently. Reviewers love a data-efficiency story with a cost narrative attached. The GPT-4 API cost for annotating 10K WMT15 De-En sentences at EAST's three-latency-level prompt is not zero — public GPT-4 pricing gives roughly *$X per 10K annotations* (the student should compute this from OpenAI's current pricing at write-up time). Ours is one full-source forward pass + `n` prefix passes on a H200, i.e., wall time not spend, and repeatable at zero marginal cost per re-annotation.

**Add to `EXPERIMENTS.md` §Primary result:** a single sentence in the abstract / intro paragraph that frames the win as "cost + quality", not just quality. The 10K-vs-660K data-size ablation stays as-is; only the framing changes.

### 7. Catchy name — 30 minutes, real impact

"Teacher-Free Read/Write Annotation" is descriptive and forgettable. EAST, DiG-SST, FAST, REINA all have three-to-five-letter acronyms. Ours doesn't.

**Candidates from the method mechanics** (backbone-derived commit-point annotation via distributional convergence):

- **TROT** — Teacher-free Read-Or-Type. Ugly.
- **SELF** — Self-annotated Efficient Latency-adaptive Fine-tuning. Overloaded.
- **DRIFT** — Distributional Read/write Inference-Free Training. Fits: our criterion measures when the predictive distribution has stopped drifting from the full-source distribution.
- **STAMP** — Self-Tagged Adaptive Machine-translation Policy. Fits: we stamp the tag on the data offline.
- **CADT** — Convergence-based Adaptive Data Tagging. Precise, hard to say.

Recommendation: **DRIFT** — matches the mechanic (predictive distribution stops drifting), and "no-drift → commit" is a memorable one-liner.

30 minutes to workshop; put on the Week-12 checklist alongside the abstract draft.

## Closest-work distinctions (deep-read, for the Related Work section)

These are the paragraphs to draft during Phase 4 writeup. Each is derived from reading the actual paper (not the abstract) so the technical distinction is precise.

### DiG-SST (Chen et al., AAAI 2024, arXiv PDF read)

**What they do.** Speech translation on MuST-C En→{De, Es, Fr}. Divergence definition: KL between `p(y_j | full audio)` and `p(y_j | partial audio)` for the next target word (§Divergence-based Policy Module). Full-vs-partial divergences are computed offline as *oracle scores*; a separate **3-layer transformer + FC policy module** is trained to *predict* the divergence from partial input alone (MSE loss). At inference: predicted divergence ≤ threshold λ → WRITE; else READ. Combined with wait-k as ceiling. Fixed λ.

**Where we differ.** Same underlying divergence idea, one stage earlier. DiG-SST predicts divergence at inference from a trained regressor — inheriting the exposure-bias problem they explicitly admit. We compute the oracle divergence *once* during data construction, use it to place `<|eor|>` / `<|eow|>` tags in the SFT data, and then throw the divergence machinery away. Inference is a plain autoregressive decoder emitting tags; no regressor, no threshold at inference, no exposure-bias gap between training-time oracle and inference-time prediction. Modality difference (speech vs text) is secondary — the mechanistic difference is *where in the pipeline the criterion lives*.

### FAST (Fu et al., EMNLP 2023, arXiv 2303.07914 — same group as EAST)

**What they do.** Speech translation. Observation: streaming encoder representations diverge from full-utterance representations, worst at the final frame (cosine similarity ~0.2 for the last frame; ~0.8 by 10 frames back). Solution: (a) **FAI** — append `m` mask tokens to the streaming input, exploit Wav2Vec2's pretraining to synthesise pseudo-future context; (b) **FAD** — distillation: teacher gets oracle future audio, student gets mask tokens, minimise KL between their encoder outputs. Combined with wait-k. No policy module.

**Where we differ.** FAST solves the mismatch at the **representation level** by manufacturing future context. We solve it at the **decision level** by moving the commit decision to training-data construction. FAST's fix still leaves a wait-k policy making the actual read/write choice; we replace wait-k with the model's own tag prediction. The mask-token trick is speech-specific (Wav2Vec2 pretraining) and doesn't port to text — but the point isn't porting, it's that they and we target orthogonal failure modes. **We should acknowledge FAST as motivating same-group work** (their EAST paper builds directly on FAST's mismatch observation) but not conflate the mechanisms.

### Local Agreement / hold-n (Polak et al., IWSLT 2022, aclanthology 2022.iwslt-1.24 — read directly)

**What they do.** Pure inference-time policies, no training, no policy module. **LA-n:** longest common prefix (LCP) of top-1 hypotheses from `n` consecutive chunks of streaming input. Commit the prefix that agrees across re-decodes. **hold-n:** commit all-but-the-last-`n` tokens of the current best hypothesis, trimming instability at the end. **SP-n:** LCP across all beam items of `n` chunks. LA-2 is the sweet spot (same trade-off as LA-n>2, cheaper). Won IWSLT 2022 medium and high latency regimes.

**Where we differ.** LA detects stability from **surface-string agreement across re-decodes**. Our criterion detects stability from **distributional convergence to the full-source distribution**. LA is model-agnostic and computationally cheap but loses signal — it treats every disagreement as equally decisive, and cannot distinguish "committed on semantically-near candidates" (should commit) from "flipped between semantically-distant candidates" (should wait). Our OT criterion has this structure by construction. LA works at inference; ours works offline at data construction, then the trained model runs plain autoregressive decoding — the LA re-decoding overhead vanishes.

**Rhetorical use.** Cite LA-n as the classical baseline for streaming-stability policies. If our OT-vs-KL ablation collapses (KL ≈ OT), we can additionally cite LA-n to say "we've shown even the crudest stability signal, given enough training-time compute, can drive competitive tag placement." That's a nice paragraph.

### REINA (Hirschkind et al., AAAI 2026 Oral, arXiv 2508.04946)

See §Blocker 2 above. **The distinction is the paper.** If we get this wrong, the paper reads as "REINA-for-text with data-construction instead of a policy head" — which is actually accurate *and* a defensible contribution, but only if we say it clearly and up front. Do not hide it in Related Work.

### CCPS — LLM confidence via representation-stability perturbations (arXiv 2505.21772, EMNLP 2025)

**What they do.** Perturb the LLM's final hidden states adversarially, measure how much the next-token distribution changes. Train a lightweight classifier to predict calibration from the perturbation-response features. Result: 55% reduction in Expected Calibration Error vs prior methods.

**Why it's on our radar.** They also measure "stability of the predictive distribution" — but under *representation perturbation*, not under *input-length variation*. Their signal is: "does this token survive a shove?" Ours is: "does this token converge as we read more source?" Different perturbations, similar spirit.

**Where we differ.** CCPS is a general-purpose confidence-calibration method, model-generic. It doesn't answer the SiMT commit question — perturbation stability doesn't tell you whether the token would change under longer source context. But CCPS *could* be an ablation baseline: replace our full-vs-partial distributional distance with CCPS's perturbation-stability score. If they match, our criterion is decomposable into a general confidence signal. If ours wins, the full-source signal carries specific information that generic confidence doesn't. **Add as a low-priority ablation** — nice for the paper, not required.

**Related literature to reference (do not deep-dive):** self-consistency (Wang et al. 2023), temperature calibration (Guo et al. 2017), semantic calibration in LLMs (Ye et al. 2025, arXiv 2511.04869). All support the general "LLMs know when they're ready" thesis. Cite for framing, not for method.

## Rejected optionals (do not do these)

- **ICLR reframing.** Wrong venue, wrong effort. Would require reframing as representation-learning / algorithmic-breadth. Don't.
- **Speech extension.** EASiST (Fu et al., AAAI 2026) already covers the speech version by the same group. Any speech move is scope creep on top of scope creep.
- **Multi-token span decisions** (commit to a phrase rather than a token). Interesting research direction, but changes the method beyond "backbone-derived tag placement." Save for follow-up work.
- **Curriculum learning on the annotation.** Unrelated to the commit criterion claim; would confound the primary A-vs-B comparison.
- **RLHF / DPO on the annotated data.** Same problem — moves the paper away from "the annotator is the contribution" and into "the training recipe is the contribution."
- **Comparing against Agent-SiMT (Guo et al. 2024b).** EAST already dismisses this baseline; re-running it costs time and adds no argument.

## Priority summary — which optionals actually get done

| # | Item | Effort | Impact | Do? |
|---|---|---|---|---|
| 1 | Scale framing (Option A) | 1 hr text | Blocker → resolved | **Yes, week 12** |
| 2 | REINA distinction subsection | 4 hr writing | Blocker → resolved | **Yes, week 13** |
| 3 | Exposure-bias dev diagnostic | 1 day compute + 1 hr text | Blocker → resolved | **Yes, week 5** (before Phase 2) |
| 4 | Reordering Figure 1 | Half day | Motivation figure | **Yes, Phase 0 upgrade** |
| 5 | 3-seed + paired bootstrap on headline | 3× SFT for primary comparison | Standard credibility | **Yes, Phase 2** |
| 6 | Data-efficiency framing | 1 hr text | Story win | **Yes, week 12** |
| 7 | Catchy name | 30 min | Memorability | **Yes, week 12** |
| REINA ablation (KL matches OT) | Already in `EXPERIMENTS.md` §Ablation grid | Amplifies REINA distinction | Already scheduled |
| CCPS ablation | 1–2 days compute | Nice-to-have | Skip unless week 11 has slack |
| 8B replication (Option B) | Post-writeup | Paper stronger | Only if Gate 3 passes with margin |

If all "Yes" items get done, the Findings blocker list is empty. The remaining risk is the Stage-I result itself — no amount of optionals fixes a null primary.

## What this document changes about the plan

Concretely, the following existing docs need touch-ups to reflect the OPTIONALS.md decisions once accepted:

- `TIMELINE.md` Phase 0 — add reordering Figure 1 as a deliverable.
- `TIMELINE.md` Phase 1 — add exposure-bias dev diagnostic as a Gate-1 add-on.
- `TIMELINE.md` Phase 2 — mark headline runs as 3-seed + paired bootstrap.
- `EXPERIMENTS.md` §Guardrails — 3-seed rule.
- `EXPERIMENTS.md` §Ablation grid — optional CCPS row.
- `CLAUDE.md` — soften "any scale" wording; add "at 2B" framing note pointing at this file.
- `LOG.md` — DECISION entry recording the scale-framing choice once made.

These are cheap edits after OPTIONALS.md is agreed. Do not make them now — read the file first, decide which items are in and which are out.
