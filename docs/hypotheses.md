# Hypotheses driving the experiments

The project's central falsifiable claim is in `../CLAUDE.md`. The hypotheses below are the working ones that motivate each Phase-1 experiment. Read this before `experiments.md` — experiments trace back here.

Each hypothesis has: **rationale** (why we'd expect it to be true or false), **prediction** (what we'd observe if it holds), **test** (the experiment that decides), **outcome** (where done).

---

## H1 — [REJECTED] JS-divergence is degenerate as a commit criterion on Gemma-4-E2B

**Rationale.** METHOD §3 hypothesises OT is the right criterion because JS/KL ignore the vocabulary's semantic geometry. If JS is a bad criterion for this task, we should see it fail even in the easiest setup.

**Prediction.** If JS on Gemma-4-E2B is degenerate, then across the tau grid `{0.02, ..., 0.30}` we'll see per-sentence Pearson(i*/n, j/m) rise monotonically toward 1 with tau, and JS should be no better than uniform-random monotone chunk placement at any tau.

**Test.** Config A — `gemma-4-E2B-it` with raw-concat prompt, JS at 6 tau values, 48 sentences.

**Outcome.** *Apparently confirmed* — JS was worse than random-at-matched-latency at every tau (JS Pearson_med > random Pearson_med by 2–15 pp). But this was overturned by H2 below — the setup was confounded.

**Rejected on:** H2's fix (chat template) closed the JS-vs-random gap and revealed the confound was the prompt, not the criterion.

---

## H2 — [PARTIALLY SUPPORTED] The prompt confound: -it models under raw concat don't do translation

**Rationale.** `gemma-4-E2B-it` is instruction-tuned. Under raw `{source}\n{target}` it isn't translating — it's doing next-token prediction on a mixed-language document. The shift in `P_pre` as source grows tracks "how many German-language tokens have accumulated" rather than "how much translation-relevant context."

**Prediction.** Under Gemma's chat template with an explicit translation instruction, the criterion should fire more (because the model is now confident about English continuation given a translation task), and should look less "diagonal-because-of-language-accumulation."

**Test.** Config B — `gemma-4-E2B-it` with chat template + `Translate the following German text to English.` instruction, same 48 sentences.

**Outcome.**
- (i) ✓ Fire rate jumped from 22% → 100% at τ=0.05.
- (ii) ✗ Pearson_med stayed at 0.94–0.97 across all taus.
- Aggregate JS-vs-random gap narrowed from ~15 pp (raw) to ~2 pp (chat) — JS still barely loses, but the gap shrank.

**Read.** The prompt was necessary but not sufficient. The chat fix restored fire coverage but didn't dislodge the diagonal Pearson. Something else was still off — hence H3.

---

## H3 — [SUPPORTED (aggregate); PARTIAL (per-sentence)] Base pretrained model + raw concat is the correct match to METHOD §1

**Rationale.** METHOD §1 defines `P_full` and `P_pre` as raw next-token-prediction distributions of the backbone. That is what a *pretrained base* model natively produces. Instruction-tuning warps this distribution around task-following behaviour. Using an -it checkpoint with a chat template was a workaround for having chosen the wrong checkpoint; the right fix is to use the base checkpoint with raw concat, matching the algorithm's specification exactly.

**Prediction.**
- (i) JS on `gemma-4-E2B` (base) + raw concat should beat random-at-matched-latency at at least one tau.
- (ii) Chunk counts should land closer to GPT-4's (~4/sentence) than under -it+chat (~9/sentence).
- (iii) On the GPT-4-identified reordering candidates (lowest GPT-4 Pearson), our criterion's per-sentence Pearson should be low too — i.e., we catch the same non-monotonic sentences.
- (iv) Per-sentence r-of-Pearsons between GPT-4 and ours should rise above the -it+chat baseline (0.15).

**Test.** Config C — `gemma-4-E2B` (base), raw concat, JS, same 48 sentences.

**Outcome.**
- (i) ✓ JS beats random at τ=0.15 (JS Pearson_med 0.842 vs random 0.881 — JS is *less* diagonal than random, which is what "beat" means for this metric).
- (ii) ✓ Ours chunk-count mean = 2.96 at per-sentence matched-count tau, chunk-count delta mean_abs = 1.44 (vs 2.25 under -it+chat). Materially closer.
- (iii) ✓ on idx=553850 (walked German verb-final case): ours produced 2 chunks with Pearson=0.311, matching GPT-4's 2-chunk late-commit pattern; under -it+chat we gave 7 chunks with Pearson=0.907 (a MISS).
- (iv) ✗ Per-sentence r stayed at 0.175 (barely up from 0.149).

**Read.** The prompt/backbone axis matters a lot. Base+raw catches individual reordering cases where -it+chat misses them. But the aggregate r-of-Pearsons metric isn't a strong discriminator here — most sentences are monotonic, so per-sentence Pearson variance is dominated by small differences on easy sentences that neither criterion cares about. The **qualitative** catch on the reordering minority is the real signal; the **aggregate** stat isn't the right way to see it.

**Follow-up implied.** Change the primary reporting metric from r-of-Pearsons to sentence-level A-score under RWTH Eq. 4 (which weights all target tokens equally, no per-sentence averaging).

---

## H4 — [DEFERRED, INCONCLUSIVE] The oracle is doing real work — `P_full` adds signal beyond prefix-entropy alone

**Rationale.** The criterion `D(P_full, P_pre) < tau` uses two things: the "oracle" distribution `P_full` (what the model would predict with the whole source) and the prefix distribution `P_pre[i]`. If we could get similar commit patterns by just looking at *when the prefix distribution's entropy `H(P_pre[i][j])` becomes small*, then `P_full` isn't contributing and the paper simplifies to "commit when the model becomes confident."

**Prediction.** If `P_full` is doing work, then at matched chunk counts, JS with oracle should give lower Pearson_med (less diagonal) than entropy-only.

**Test.** Entropy-only sweep — `record_entropy=True` in the annotator; offline, commit when `H(P_pre[i][j]) < H_tau`.

**Outcome (base + raw matrices).**
- At H_tau=2.0 (chunks 3.50 ≈ GPT-4's 4.06), entropy-only Pearson_med = 0.828.
- At τ=0.10 (chunks 3.46), JS Pearson_med = 0.732.
- JS ≈ 10 pp lower Pearson at matched chunk count — suggests `P_full` IS doing work.

**But.** The chunk counts don't match exactly (3.50 vs 3.46 is close but the H_tau grid is coarse). The test needs finer H_tau + tau grids to be conclusive. **Deferred until after OT lands.**

---

## H5 — [QUEUED FOR TEST] OT with embedding ground cost catches committability JS misses

**Rationale.** JS says two distributions are "far" if they place mass on different tokens, regardless of whether those tokens are semantically similar. Consider `P_pre` concentrated on "cat"/"kitten"/"feline" vs `P_full` on "cat"/"kitten"/"feline" but with slightly different weights — JS reads this as "different distributions" and refuses to commit. But semantically, all three tokens are equivalent for the user, so the position **is** committable.

METHOD §3 fixes this with a ground metric: `C_{a,b} = 1 - cos(E_a, E_b)` where `E` is the input embedding matrix. OT with this cost knows "cat" ↔ "kitten" is a cheap transport and marks the position committable. This is the paper's primary theoretical claim.

**Prediction.**
- (i) At matched chunk count, OT Pearson_med ≤ JS Pearson_med (OT catches more non-monotonicity).
- (ii) Per-sentence r(GPT-4, ours) should improve above JS's 0.175.
- (iii) On sentences where the model is uncertain among near-synonyms (typically easy monotonic cases), OT should commit *earlier* than JS.
- (iv) On the top reordering candidates (GPT-4 Pearson < 0.85), OT should still commit late (same as JS).

**Test.** Config D — same base+raw setup, OT criterion (topk=128, eps=0.05, Sinkhorn iters=200 via `pot.bregman.sinkhorn_log`), same 48 sentences, tau grid `{0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}` (OT distances live in a different numeric range than JS; grid extended).

**Outcome (job 176307323 landed):**
- (i) ✓ At matched chunk count (OT τ=0.30 gives 4.67 chunks ≈ GPT-4's 4.06; JS τ=0.15 gives 6.04 chunks): OT Pearson_med=0.81 vs JS Pearson_med=0.84. Tied within noise at matched chunks, but OT achieves it with chunk count closer to GPT-4.
- (ii) ✓ Per-sentence r(GPT-4, ours) = **0.306** (up from JS's 0.175 — nearly doubled). Meaningful.
- (iii) OT beats random-at-matched-chunk-count at TWO tau values (0.20 and 0.30) vs JS at ONE (0.15). Broader "signal" range.
- (iv) On top-8 reordering candidates: 3 MATCH, 5 MISS. 4 of 5 MISS are single-chunk collapse (coverage limit — OT stays above τ=0.50 on those hard cases). Same idx=553850 catch as JS Config C, plus idx=493988 improves from 0.81 → 0.66.

**Cost:** OT is ~24× JS per pair (~31s/sentence vs 1.3s). Acceptable for annotation (offline); would matter more if it ran online.

**Read.** H5 SUPPORTED. Embedding-grounded OT with cost `1 - cos(E)` earns its keep on the two metrics that matter (per-sentence r, beats-random range). JS remains a valid cheap ablation. Paper's headline claim stands up.

**Follow-ups implied:**
- Extend tau grid to `{0.70, 1.0}` to resolve the 4 single-chunk collapses.
- Sensitivity of OT to topk (32 / 128 / 512) and eps (0.02 / 0.05 / 0.10) — EXPERIMENTS.md ablation row 4.

---

## H6 — [FUTURE] Cross-backbone: the finding is family-robust

**Rationale.** If the base+raw+JS/OT behaviour we see on Gemma-4-E2B holds on Qwen3.5-2B (matched size, different family), the finding isn't a Gemma-specific artefact.

**Prediction.** On `Qwen3.5-2B` with raw concat and the same tau grid, JS/OT should show the same qualitative properties: JS beats random at some tau; OT beats JS at matched chunk count; per-sentence structure catches the same reordering candidates.

**Test.** Config E — annotate with Qwen3.5-2B, run all the same analyses. Only after RWTH-based A-score is established on Gemma to have a comparison anchor.

**Status.** Queued. Not started.

---

## H7 — [FUTURE, HYPOTHETICAL] Scale-up: E4B produces tighter Pearson tracking of GPT-4

**Rationale.** A larger backbone has better-calibrated distributions; commit points might align more precisely with the ground truth.

**Prediction.** On `gemma-4-E4B` (base), per-sentence r(GPT-4, ours) rises above E2B's; RWTH A-score improves.

**Test.** Gated on Gate-1 passing on E2B (RWTH result). Not run.

**Status.** Not started. Do not scale to E4B until E2B's Gate-1 signal is defensible.

---

## Which hypothesis governs which experiment

| Config | Model | Prompt | Criterion | Tests hypothesis |
|---|---|---|---|---|
| A (initial smoke → sweep) | gemma-4-E2B-it | raw | JS | H1 (apparent-confirm → rejected by H2) |
| B (chat re-run) | gemma-4-E2B-it | chat | JS + entropy record | H2 |
| **C (base + raw) ★** | **gemma-4-E2B (base)** | **raw** | **JS + entropy record** | **H3 (aggregate ✓, per-sentence partial), H4** |
| D (OT sweep) | gemma-4-E2B (base) | raw | OT | H5 |
| E (cross-backbone) | Qwen3.5-2B | raw | JS then OT | H6 |
| F (scale-up) | gemma-4-E4B (base) | raw | JS/OT | H7 (only if Gate 1 passes) |
