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
Update 2026-08-18: Gate 1 passed at n=210 stratified (OT reordering MATCH 54.3% > monotone 38.6%). E4B base downloaded and SFT cond-A queued (job 176530894). E4B annotation cond-B running (176530895).

---

## H8 — [CONFIRMED at n=10K] OT-annotated training data teaches better streaming translation than GPT-4-annotated data (H5 → SFT descendant)

**Rationale.** H5 showed OT catches non-monotonicity where JS misses. If that annotation quality matters for the DOWNSTREAM policy, then models fine-tuned on OT-annotated data should produce better translations under streaming inference than models fine-tuned on GPT-4-annotated data. Cond-A's uniform 4-6-word GPT-4 chunks may over-fit to a specific chunking rhythm; cond-B's variable-length chunks (including 28% single-chunk collapse for reordering-heavy sentences) should generalize better to arbitrary commit positions imposed by streaming policies.

**Prediction.**
- (i) Under a fixed-latency streaming policy (wait_k), cond-B should give strictly higher BLEU than cond-A at matched AL.
- (ii) The gap should hold across the useful wait-k range (k ∈ {3, 5, 7}).
- (iii) Offline BLEU (no streaming) should be UNCHANGED — cond-B shouldn't degrade translation quality without streaming.

**Test.** Config G — matched cond-A and cond-B trained on same 9,567 latency-balanced sentences, both under identical SFT recipe (Gemma-4-E2B base + extended tokenizer + early stopping + lr 2e-5, effective batch 16). Streaming eval on newstest2013 (3,000 sents) under wait_k ∈ {3, 5, 7} + check_argmax.

**Outcome.**
- (i) ✓ At each wait-k, cond-B strictly beats cond-A:
  - wait_k=3: cond-A 16.49 BLEU @ AL 2.10, cond-B 22.14 BLEU @ AL 2.35. **Δ +5.65 BLEU.**
  - wait_k=5: cond-A 21.53 BLEU @ AL 3.17, cond-B 26.94 BLEU @ AL 3.54. **Δ +5.41 BLEU.**
  - wait_k=7: cond-A 23.61 BLEU @ AL 4.19, cond-B 28.40 BLEU @ AL 4.64. **Δ +4.80 BLEU.**
- (ii) ✓ Signal is uniform across the wait-k range. Consistent, not noise.
- (iii) ✓ Offline BLEU: cond-A 32.41, cond-B 32.54. Statistically identical.

**Read.** H8 CONFIRMED at n=10K, matched conditions, matched sentences. The paper's headline result.

**Follow-ups implied.**
- Extended wait-k to k ∈ {1, 9, 11} for a smooth trade-off curve (jobs 176531163/164 queued).
- Per-latency-prompt (low/high) — EAST Table 3 mirror (jobs 176531165/166 queued).
- Cross-backbone (H6, Qwen3.5-2B) replication in flight — does the gap hold on a different family?
- Scale-up (H7, Gemma-4-E4B) in flight — does the gap hold at 2× params?
- Data-scale curve on champion (10K → 50K).

---

## H9 — [REFUTED at n=10K] Model-driven adaptive commitment (check_argmax) can produce reasonable AL/BLEU trade-off without external policy

**Rationale.** If cond-B has learned to represent commit points as EOR tokens in its output distribution, then a "let the model decide" policy — at each source position, emit EOR if that's the model's argmax; else feed the next source word — should produce a natural streaming behaviour without needing wait-k. Cond-B in particular saw single-chunk-collapse rows in training (the "commit at end" case), so its argmax at end-of-source and at natural pause points should be EOR.

**Prediction.** Under `check_argmax`:
- (i) cond-B should give chunks/sentence > 1 (model voluntarily commits somewhere mid-source).
- (ii) cond-B's AL should be in the useful range (< 10).
- (iii) BLEU should be competitive with wait_k=5-7.

**Test.** `src/eval/extrinsic.py --mode streaming --policy check_argmax` on newstest2013 (3,000 sents) for both arms.

**Outcome.**
- (i) ✗ **Both models emit chunks/sentence = 1.00.** Neither cond-A nor cond-B voluntarily emits EOR mid-source. At every intermediate position, the model's argmax is "next source word", never EOR.
- (ii) ✗ AL = 18.20 (essentially offline) for both — model always reads all source before speaking.
- (iii) BLEU 30.66 / 30.76 for A/B — comparable, at maximum latency.

**Read.** H9 REFUTED at n=10K. Under threshold-argmax policy, both models revert to offline-like behaviour. SFT on the EAST format at 10K rows is not enough to teach the model to CHOOSE commit points via argmax alone — it learns the tag as a next-token in the training pattern `<latency> src <eor> tgt <eow> src ...`, not as an autonomous policy decision.

**Consequence for the paper narrative.** The claim isn't "cond-B learned when to commit" (H9 refuted). The claim is H8: "cond-B produces higher-quality translations under any imposed streaming latency." The mechanism ships as: OT annotation quality → better generalization to fixed-latency policies. Wait-k is the natural way to demonstrate the effect.

**Open question.** Does check_argmax start working at larger data scales (50K, 660K) or on larger backbones (E4B, 9B+)? The queued scale-up runs will inform this.

---

## H10 — [QUEUED] Annotator quality is model-invariant (cross-annotator SFT ablation)

**Rationale.** In our default recipe, the same backbone that annotates the training data also gets fine-tuned on it (matched: annotator = SFT model). This entangles two things:
(a) intrinsic quality of the annotator's chunk placements ("does the annotator identify positions where a translator can commit safely?");
(b) matched-representation advantage ("does SFT work better when its training data uses commit points its own embeddings agree with?").

If (a) dominates, chunks derived from any competent annotator should improve any SFT backbone. If (b) dominates, annotator-SFT pairs must be matched or the transfer degrades. Answer determines whether "our annotator is universal" or "our annotator + SFT is a coupled system."

**Prediction.**
- (i) E4B-annotator → E2B-SFT should be within 1-2 BLEU of E2B-annotator → E2B-SFT (larger annotator → smaller SFT should transfer well; better chunks generalize down).
- (ii) E2B-annotator → E4B-SFT may show slight LIFT over E4B-annotator → E4B-SFT (a smaller annotator's chunks may transfer up cleanly OR may underspecify — need data).
- (iii) Cross-family (Gemma ↔ Qwen) should be the harshest test: if their embedding spaces disagree on token-neighborhood structure, cross-family transfer should degrade more than within-family (E2B ↔ E4B).

**Test.** After all three annotations complete (E2B ✓, E4B in flight, Qwen in flight), build 3 cond-B datasets (one per annotator) and run 6 off-diagonal SFTs. Streaming eval at wait_k=5 to keep the sweep tractable. See `07-next_steps.md` §10.

**Status.** Queued. Not started.

**Consequence.** If H10 confirmed, the annotator is a universal preprocessing step — you can annotate once with the largest available backbone and reuse for any SFT. If refuted, the annotator ships as a per-backbone artifact (bigger deployment cost, weaker paper story).

---

## Which hypothesis governs which experiment

| Config | Model | Prompt/Setup | Criterion | Tests hypothesis |
|---|---|---|---|---|
| A (initial smoke → sweep) | gemma-4-E2B-it | raw | JS | H1 (apparent-confirm → rejected by H2) |
| B (chat re-run) | gemma-4-E2B-it | chat | JS + entropy record | H2 |
| **C (base + raw) ★** | **gemma-4-E2B (base)** | **raw** | **JS + entropy record** | **H3 (aggregate ✓, per-sentence partial), H4** |
| D (OT sweep) | gemma-4-E2B (base) | raw | OT | H5 |
| F (Gate 1) | gemma-4-E2B (base) | raw, n=210 stratified | OT, JS | H5 aggregate (OT passes; JS fails) |
| **G (SFT matched) ★★★** | **gemma-4-E2B (base)** | **SFT n=10K, matched A/B** | **streaming eval** | **H8 CONFIRMED, H9 REFUTED** |
| Qwen replication | Qwen3.5-2B | matched A/B | streaming eval | H6 (in flight) |
| E4B replication | gemma-4-E4B (base) | matched A/B | streaming eval | H7 (in flight) |
| Scale-data | champion | n=10K/20K/30K/40K/50K matched | streaming eval | H8 at scale (queued) |
| Cross-annotator | E2B, E4B, Qwen (6 off-diagonal SFTs) | matched B, mismatched A | streaming eval | H10 (queued) |
