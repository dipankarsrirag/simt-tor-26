# Theory — why the method should work, and what "self-supervision" is doing

**Status:** open questions. This doc is the starting brief for a session focused on
making the method mathematically sound. Everything below is either informal
handwaving that needs formalising, or explicit gaps flagged for the next session.

Read `_archive/docs/method-formal.md` first for the algorithm as implemented. Read
`docs/method.md` for the operational shape. Then this doc.

---

## 1. What we're actually claiming

The pipeline, in one sentence: **at data-construction time we hold the full source
`S`; for each target token `y_j` we find the earliest source prefix length `i*_j`
such that the backbone's next-token distribution conditioned on the prefix has
converged to the distribution conditioned on the full source; we insert
`<|end-of-read|>` at those positions and fine-tune on the interleaved sequences.**

Formally, the commit rule is

$$i^*_j \;=\; \min\bigl\{\, i \in \{1,\dots,n\} \;:\; D\bigl(P_M(\cdot \mid S,\, T_{<j}),\; P_M(\cdot \mid S_{\le i},\, T_{<j})\bigr) < \tau \,\bigr\}$$

(fall back to `n` if the criterion never fires; then monotonise
`i^*_j := max(i^*_j, i^*_{j-1})`).

Fine-tuning then targets

$$\mathcal{L}_{\text{SFT}}(\theta) \;=\; -\,\mathbb{E}_{(S,T,\mathbf{i}^*)}\!\left[\,\log p_{M_\theta}\!\bigl(\text{interleave}(S, T, \mathbf{i}^*, \text{EOR}, \text{EOW})\bigr)\,\right]$$

with the loss reduced over the entire interleaved sequence — source tokens, target
tokens, and the read/write special tokens.

**The claim we need to earn:** minimising this loss yields an inference-time policy
that, when acting autoregressively on streaming source at test time (no access to
`S` beyond what has been read), makes *appropriate* commit decisions.

That word "appropriate" is where the theoretical work lives.

---

## 2. The core question: what is this actually taking supervision from?

The pipeline never sees a human-annotated read/write boundary. There is no gold
policy. What we have is:

1. `M` itself, in an oracle configuration that has access to `S`.
2. A distance `D` between distributions.
3. A threshold `\tau`.

The commit points `i^*_j` are a function of `(M, S, T, D, \tau)`, nothing else. Then
we ask `M` (post-fine-tune) to reproduce those decisions from streaming input alone.

Two ways to frame what's going on:

### Frame A — Self-distillation from an oracle head

The "teacher" is `p_M(\cdot \mid S, T_{<j})` — the full-source head of the
*same model*. The "student" is `p_M(\cdot \mid S_{\le i^*_j}, T_{<j})` — the
partial-source head, also the same model. The commit criterion asks: at what
prefix length has the student already agreed with the teacher, i.e. at what
prefix length is the oracle information about `S_{>i}` no longer changing the
next-token belief?

Fine-tuning then makes two commitments to the model:

- **(a)** at prefix `i^*_j` the target token `y_j` is a good prediction (loss on
  target tokens);
- **(b)** at prefix `i^*_j` the appropriate next action is to emit EOR, and at
  prefix `i^*_j - 1` (or wherever the previous chunk ended) it is not (loss on
  the EOR/EOW tokens).

The self-distillation frame is honest about what "teacher-free" means: we are
distilling from a **teacher-forced full-source oracle** into a **prefix-conditioned
autoregressive policy**, both instantiated by the same weights.

### Frame B — Learning a stopping rule from a Bayes-optimal segmentation

The commit point `i^*_j` is the smallest prefix at which, under `M`, the
posterior over the next target token is $\tau$-close to what it would be with
full information. Under sufficient conditions on `M` and `D` (see §3), this
is (i) monotone in `i`, (ii) attained at a well-defined stopping time.

The fine-tuning objective then teaches `M` to *recognise* that stopping time
from streaming input — i.e. to learn the map $(S_{\le i}, T_{<j}) \mapsto \{$commit
now, keep reading$\}$ that reproduces `i^*_j`.

This frame invites the analogy to sequential hypothesis testing (Wald 1947).
The commit point is a first-passage time; fine-tuning is learning to approximate
a first-passage time from partial-history features.

---

## 3. Assumptions we're leaning on (mostly implicit — we need to name them)

Every argument for "why this should work" leans on assumptions about `M`, `D`, and
`\tau`. The current codebase makes several of these **implicitly**. Making the
method mathematically sound means listing them, checking each, and knowing which
break.

### A1 — Monotonicity in expectation of `D(P_full, P_pre[i])` in `i`

**Statement.** For a fixed `(S, T, j)`, the divergence
$D_{ij} := D(P_M(\cdot|S, T_{<j}),\, P_M(\cdot|S_{\le i}, T_{<j}))$ is
non-increasing in `i` **in expectation** over "reasonable" `(S, T)` pairs.

**Why we need it.** Without monotonicity, the first-crossing time
$\min\{i : D_{ij} < \tau\}$ is not a meaningful stopping rule — it may "commit,
uncommit, recommit" as more source is read, and the code post-hoc enforces
monotonicity with `i^*_j := max(i^*_j, i^*_{j-1})`. That enforcement is a patch
over a theoretical hole.

**Why it should hold.** More source is more information; for a well-calibrated
`M`, more information should on average bring `P_pre[i]` closer to `P_full`.
Formally this is a data-processing-inequality style statement, but it only
applies to the true posteriors, not to a fixed model's approximation of them.

**Empirical status.** Not measured. Should be trivial to add a check:
per-sentence, per-target-token, plot `D_{ij}` as a function of `i` and count
non-monotone transitions. If they're rare (<5% of tokens), we can defend the
monotonisation as a light post-hoc smoothing. If they're common (>20%), the
whole "first-crossing time" framing needs reworking.

### A2 — `D` is a legitimate divergence for this purpose

Current choice: entropic-regularised OT with cost `C_{ab} = 1 - \cos(E_a, E_b)`
over top-`k = 128` support (see `_archive/docs/method-formal.md` §3).

**Why OT and not KL.** KL treats "cat" vs "dog" as exactly as wrong as "cat" vs
"the". OT with an embedding-grounded cost knows the first pair is close. The
implicit assumption is that the model's embedding geometry is a meaningful proxy
for semantic distance — that "committing while uncertain between semantic
near-neighbours" is safe, and "committing while uncertain between semantic
far-neighbours" is not.

**What breaks if the embedding geometry is bad.** If `E_{\text{cat}}` and
`E_{\text{the}}` happen to be nearby in the model's input embedding space (which
does happen for stopwords and small-magnitude embeddings), OT ceases to
distinguish safe from unsafe uncertainty.

**Empirical anchor.** Phase-1 config-D-ext (OT + `\tau = 0.30`) beats
JS-divergence + `\tau = 0.05` on the reordering subset (see
`_archive/docs/phase1-annotator-experiments.md`). That's evidence OT is doing
some real work vs. a pure-distributional distance. But we haven't shown OT is
*right*, only that it beats an obvious foil.

**What to formalise.** Under what conditions on `E` does the OT criterion
reduce to something more familiar? Is there a `\tau`-scaling relationship
between OT and KL under Gaussian embedding assumptions?

### A3 — The teacher-forced full-source oracle is close to the free-generation full-source oracle

At data-construction time we teacher-force on `T` when computing both
`P_full[j]` and `P_pre[i][j]`. The tags we place are conditioned on this
teacher-forced trajectory. At inference `M` generates target tokens
autoregressively, not from the reference; the trajectory it walks is different.

**Why we need the assumption.** If teacher-forced `P_M(\cdot | S, T_{<j})` is
very different from `P_M(\cdot | S, y_1..y_{j-1})` where `y_{<j}` was freely
generated, then the commit points are calibrated to a distribution the model
never sees at inference.

**Why it should hold, weakly.** For a well-trained MT model, if the target is
a good translation of the source, the free-generation trajectory should stay
close to the teacher-forcing trajectory in the same distributional
neighbourhood. But this is exposure bias in a new dress. Standard results
(Ranzato et al. 2016 on scheduled sampling; Bengio et al. 2015) tell us
teacher-forcing and free-generation *do* diverge in autoregressive models.

**Empirical anchor.** SFT training itself operates in the teacher-forcing
regime, but is loss-reduced also over the tag positions — which means the model
does at least see the correct EOR/EOW placement relative to teacher-forced
targets. The extent to which this transfers to free generation is an empirical
question. WMT15/22 BLEU numbers are the aggregate answer.

### A4 — `\tau` is transferable across sentences

We pick `\tau = 0.30` on a small dev-side sweep (Gate-1, 210 sentences) and
apply it to all 78K training rows. Implicit: the commit threshold is a
*global* hyperparameter, not sentence- or context-dependent.

**Why this might be wrong.** A short sentence with easy source-target
alignment has different distributional dynamics than a 40-token verb-final
German sentence. A single `\tau` that fires appropriately on the former may
fire too aggressively (or never) on the latter.

**What we do about it currently.** The fallback ladder `\tau_ladder = [0.30,
0.50, 0.70, 1.00]` in `scripts/08_build_sft_dataset.py::commit_with_fallback`
handles the "never fires" case by relaxing `\tau` until at least 2 chunks
emerge. This is a per-sentence adaptivity — but only a coarse one, and only
in one direction (relax, never tighten).

**What to formalise.** Is there a principled per-sentence `\tau` derivable
from a statistic of `(S, T)`? Sentence length? Reordering severity (which we
already measure via Pearson-of-alignment for Gate 1)? An entropy statistic of
`M`?

### A5 — The chunking rule (consecutive shared commit points) is optimal

Once we have `\{i^*_j\}_{j=1}^m`, the code groups consecutive `j`'s that share
the same `i^*_j` into one write chunk (`_chunks_from_commit` in
`src/annotator/annotate.py`). This is procedural, not derived from anything.

**Alternative chunkings to consider.** Merge chunks with < K source words
(EAST §3.1 does this, `min_src_words=4` in our config). Merge across chunk
boundaries where the target is a syntactic constituent. Split chunks that
straddle a target discourse boundary.

**What to formalise.** Under what objective (BLEU? training-loss-per-token?
inference-latency?) is "one chunk per unique commit point" optimal? Could a
different grouping (e.g. dynamic programming over an objective that trades
off commit granularity vs. latency) systematically beat the current rule?

---

## 4. Related theoretical frames we should place ourselves in

### 4.1 Sequential hypothesis testing / optimal stopping

The commit point is a first-passage time of the process
$D_{ij}$ across the threshold `\tau`. Wald's sequential probability ratio test
(SPRT) framework studies exactly this kind of "keep observing until confidence
crosses a threshold" problem. Our setup differs in that the process is a
distributional divergence, not a likelihood ratio — but the mathematical
machinery of optimal stopping (Bellman, Peskir & Shiryaev 2006) applies.

**Question for the next session.** Is our `\tau`-crossing rule Bayes-optimal
under any natural loss function trading off latency and translation quality?
If yes, that's a clean theoretical positioning. If no, what is the gap?

### 4.2 Self-training and pseudo-labelling

The "teacher = M with oracle info, student = M with streaming info" frame
puts us squarely in self-training territory (Yarowsky 1995; Xie et al. 2020
noisy student; more recent LLM self-improvement work). The distinction is
that the teacher and student are literally the same parameters, differing
only in what they condition on. That is unusual; most self-training uses two
model instances or two model snapshots.

**Question for the next session.** Is there a self-training convergence
theorem that applies? Under what conditions does the fine-tuned model's
partial-source policy converge to the oracle's full-source-implied policy?

### 4.3 Reward learning from demonstrations / imitation learning

Read/write policies are decision processes. Our procedure amounts to
generating (state, action) trajectories — where state is `(S_{\le i}, T_{<j})`
and action is `{read, write}` — from an "expert" (the oracle head), then
training the model to imitate. That is behaviour cloning.

**Question for the next session.** Behaviour cloning suffers from
compounding error under distribution shift (Ross & Bagnell 2010, DAgger).
Do we hit this? Is there evidence in AL vs BLEU curves that streaming
inference drifts away from the training distribution?

### 4.4 Information bottleneck

The commit criterion says "commit when reading more source no longer changes
the target distribution". That is precisely an information-bottleneck
statement: `S_{>i}` becomes conditionally uninformative about `y_j` given
`S_{\le i}` and `T_{<j}`. Tishby's IB framework (1999; Alemi et al. 2017 for
deep learning) is the right formalism to reach for.

**Question for the next session.** Can we state the commit criterion as an
IB constraint, and derive `\tau` from an information-theoretic principle
rather than tuning it on dev data?

---

## 5. What the next session needs to produce

**Deliverables, in priority order:**

1. **Explicit assumption list** (formalise A1–A5 above, add any missed).
2. **Empirical checks** for each assumption — small scripts, on the existing
   matrices files, that measure the assumption's failure rate on real data.
3. **Positioning statement** — which of the frames in §4 does the method sit
   in? Are we doing sequential hypothesis testing, self-training, imitation
   learning, or IB — and which one gives the cleanest theoretical guarantee?
4. **Minimum-viable-theorem** — pick the ONE thing we can state formally and
   prove (or at least argue rigorously). Candidates:
   - Under A1–A3, the fine-tuning loss upper-bounds a proxy for
     $\mathbb{E}[|p_M^{\text{stream}}(\cdot) - p_M^{\text{oracle}}(\cdot)|]$.
   - Under A2 + specific embedding assumptions, our OT criterion is
     equivalent (up to `\tau`-scaling) to KL — testable empirically.
   - Under an IB framing, `\tau` corresponds to a specific point on the
     rate-distortion curve; we identify which.
5. **Paper §Method 2.0** — rewrite `_archive/docs/method-formal.md` to
   include the theoretical grounding, not just the algorithm.

**What NOT to do in the next session:**

- Debate whether the empirical results validate the method (they do enough
  — see `docs/experiments.md` for the numbers).
- Rewrite the annotator implementation (it works; the theory is what needs
  rewriting).
- Chase a full-generality proof if a well-scoped assumption-heavy one lands.

---

## 6. Handoff pointers

Read in this order:
1. `_archive/docs/method-formal.md` — algorithm, precisely, 7-step pipeline.
2. `docs/method.md` — implementation-shaped version + the seven-step walk-through.
3. `_archive/docs/random-floor-and-ot.md` — worked examples for the OT criterion.
4. `docs/hypotheses.md` — the four paper-facing hypotheses (P1–P4).
5. This doc.
6. `CLAUDE.md` — the project's non-negotiable invariants.

**Code entry points for empirical checks:**
- `src/annotator/annotate.py::annotate_pair` — where the divergence matrix is computed and commit points are chosen.
- `src/annotator/criterion.py` — the divergence implementations (OT, JS, KL, entropy).
- Existing matrices files:
  `results/annotate/gemma-4-E2B-it/{pair}/matrices.jsonl` (Gemma-2B),
  `results/annotate/Meta-Llama-3-8B-Instruct/{pair}/matrices.jsonl` (Llama-3-8B),
  `results/annotate/gemma-4-E4B-it/{pair}/matrices.jsonl` (Gemma-4B).
  Each record has the full `n × m` divergence matrix; empirical checks of A1
  (monotonicity in `i`) can run offline on these without any GPU.

**Related-work anchors already read:**
- EAST (Fu et al. ACL Findings 2025) — GPT-4-based chunking; we replace GPT-4
  with the backbone.
- Wang et al. 2024 "Conversational SimulMT" (arXiv 2402.10552) — fastalign
  chunking, dialogue format, RALCP inference; see LOG.md 2026-08-31 entry.

**Anti-goals for the next session:**
- Don't introduce a second model as a genuine teacher — that breaks the
  "teacher-free" identity of the paper.
- Don't require any signal at inference beyond streaming source + past
  target tokens — that breaks Invariant #2 in `CLAUDE.md`.
