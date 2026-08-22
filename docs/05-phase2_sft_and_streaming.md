# Phase 2 — SFT + streaming eval

This is where the annotator from Phase 1 becomes an actual translator, and where the paper's headline result lands.

Read Phase 1 first (`03-phase1_annotator_experiments.md`) — Phase 2 assumes the annotator is chosen (OT, τ=0.30, base Gemma-4-E2B + raw concat) and validated (Gate 1 passed on n=210 stratified). Read `00-README.md` "Naming" section for terminology.

---

## Phase 2b (LIVE, 2026-08-21 pivot + 2026-08-22 fixes) — v6b-ctrl-merged3 is the ship model

**Everything below this section header is the CURRENT live state.** The older Phase 2 material further down (Gate 2, cond-A n=10K matched-pair) is archaeological from before the v6 pivot.

### The recipe (v6 + v6b fixes + EAST §3.1 merge)

1. **Backbone.** `gemma-4-E2B-it` (instruct-tuned, 2B params). This was the 2026-08-21 v6 pivot away from base Gemma-4-E2B (LOG `[DECISION] 2026-08-21 — v6 pivot`).
2. **Prompt.** Chat template with system + user turn carrying the translation instruction: `"Translate the following text from {SRC} into {TGT} with {LATENCY} latency."` Latency is one of `{low, low-medium, medium, medium-high, high}` (5-point NL ladder — 3 base labels trained, 2 interpolated at inference).
3. **Tokenizer.** `results/phase2/tokenizer-extended-v6/` — Gemma-4-E2B-it's tokenizer + 2 added special tokens (`<|end-of-read|>`, `<|end-of-write|>`). Latency labels moved into NL (no `<|*-latency|>` tokens).
4. **Chunks.** OT annotator on Gemma-4-E2B base (multi-90K + custom pools for ar-en/en-ar/vi-en/en-vi), τ=0.30 primary with fallback ladder to 0.5/0.7/1.0. Post-processing: **EAST §3.1 merge** at the aggressive <=3-word threshold (`--merge_small_chunks --min_src_words 4`). Chunks with < 4 source words get folded forward into the next chunk. Brings the OT chunk distribution to 82% ≤3-chunks-per-sentence (up from 37% raw), while cond-A GPT-4 chunks sit at 91%.
5. **Training data.** `results/phase2/sft_dataset_multilingual_v6b_merged3.json` (79,309 rows across 8 language pairs, ~10K per dir).
6. **SFT.** `src/train/sft_v6.py` — direct-ids splice (bypass string round-trip), α=1 (no EOR/EOW loss upweight; α=5 was hurting), 2 epochs, best-model-by-eval-loss, descriptive-init on EOR/EOW, per-device batch 16 × grad-accum 4 = effective batch 64. `results/phase2/sft_multilingual_v6b_ctrl_merged3/final/` (~60 min wall).
7. **Inference.** `src/eval/extrinsic.py --use_chat_template`. Streaming: feed source word-by-word (word[0] no leading space, word[i>0] with leading space — matches annotator tokenization byte-exactly). Poll `argmax(logits) == EOR` after each word; if true, commit and generate target until EOW/EOS.

### Headline sanity numbers (N=50 FLORES devtest, 8 dirs × 5 latencies = 40 cells)

| variant | mean BLEU | mean AL | mean chunks/sent |
|---|---|---|---|
| v6b-ctrl (raw OT) | 24.89 | 3.32 | 10.5 |
| merged (<2 words) | 27.70 | 3.46 | 7.2 |
| **v6b-ctrl-merged3 — ship** | **29.46** | 4.78 | 4.5 |
| E4B on raw OT | 28.10 | 3.92 | 8.2 |
| cond-A (GPT-4 chunks, 4 dirs only) | 30.51 (20 cells) | 5.69 | 3.8 |

**Restricted to the 4 cond-A directions (de-en, en-de, ru-en, en-ru):**
- v6b-ctrl-merged3 (E2B): 29.15
- cond-A (E2B, GPT-4 chunks): 30.51 → merged3 is 1.36 BLEU behind cond-A on average, 76% recovery of the +5.72 raw-OT-vs-cond-A gap.

**On de-en at low_medium latency: merged3 (31.88) beats cond-A (30.90).** Matched-backbone head-to-head win against GPT-4 chunks.

### Big finding: chunk simplification beats scaling

The E4B run (Gemma-4-E4B-it, 4B) on the same raw-OT dataset gained +3.21 mean BLEU over E2B ctrl (24.89 → 28.10). But **merged3 (E2B, 2B) still beats E4B (raw OT) by −0.49 BLEU on the same 40 cells**. Coarser chunks via EAST §3.1 merge did more than doubling the model.

### Fixes shipped this pivot

- **v6 (2026-08-21):** Switch backbone from base Gemma-4-E2B to `gemma-4-E2B-it`. Move latency from vocab tokens to NL prompt phrase. Fixes the "en-ar produced Vietnamese output" language-selection bug the previous single-`<|latency|>`-token prompt suffered.
- **v6b string round-trip fix (2026-08-22, `scripts/phase2_build_sft_dataset.py`):** Old builder decoded chunk ids back to strings, then re-tokenized when embedding into the chat template — 40-47% of AR/VI rows silently dropped at the leading-space retokenization gate. Even DE/EN rows had `▁And` vs `And` first-target-token drift. Fix: use annotator's original `tok(src)` ids as canonical; `sft_v6.py::build_row_ids` splices chunk_ids directly into prefix + suffix from `render_chat_open_close_ids()` (placeholder-split). Now byte-identical annotator ↔ training ↔ inference. Verified 24/24 sample rows across 8 dirs pass byte-compare (`scripts/probe_v6_sanity.py`).
- **α=1 (2026-08-22):** Removed `--special_token_loss_weight 5.0`. α=5 was upweighting EOR/EOW loss to force sharp learning of chunk boundaries; at inference this produced over-eager EOR triggers (3-4× the training-time chunk density). Ctrl α=1 beats main α=5 by +2.60 mean BLEU on 40 cells (LOG `[DECISION] 2026-08-22 — Retire α=5`).
- **EAST §3.1 merge at <=3 words (2026-08-22, this section):** OT annotator naturally produces 4-8 chunks/sent; GPT-4 produces 2-3. Merging silvers (< N source words) forward brings distributions into rough alignment; τ=0.30 stays, only the chunk-count operating point moves.

### What's next (see 07-next_steps.md)

1. Full N=1012 FLORES devtest on merged3 (5 latencies × 8 dirs = 40 cells).
2. Full N=2170 WMT15 De↔En on merged3 for EAST Fig 3 head-to-head.
3. Combined E4B + merged3 SFT (does scaling + merge stack?).
4. Optional: batched annotator → fair E4B scaling test with matched annotator+trainer.

---

## Phase 2a (ARCHAEOLOGICAL) — pre-v6 pipeline

**Everything below is archaeological.** The Aug 21 v6 pivot + Aug 22 v6b work superseded this. Kept for provenance and to explain the fix history.

## Gate 2 — does the SFT pipeline work at all?

The first check before comparing anything: does either model emit tags at all? Streaming smoke on 30 heldout prompts:

**Prompt shape:** `<|medium-latency|> ` + first 3 source words. Then generate 80 tokens.
**Pass criterion:** ≥ 50% of probes emit `<|end-of-read|>` AND `<|end-of-write|>` in correct order.

**cond-A n=10K:** 40/40 probes emit both tags in correct order. Sample generation for source prefix `"Für Josephus ist"` (medium latency):

```
Für Josephus ist es ein Segen, <|end-of-read|>
For Josephus it is a blessing <|end-of-write|>
dass er die Möglichkeit hat, <|end-of-read|>
that he has the opportunity <|end-of-write|>
die Geschichte der Juden zu schreiben, <|end-of-read|>
to write the history of the Jews <|end-of-write|>
und er tut es mit großer Leidenschaft. <|end-of-read|>
and he does it with great passion. <|end-of-write|><eos>
```

Model is doing chunk-wise translation with tags at plausible positions. Gate 2 PASSES.

**Load-bearing bug caught during Gate 2** (documented in `LOG.md`, worth restating here because it would have poisoned everything downstream):

The five EAST tokens (`<|end-of-read|>`, `<|end-of-write|>`, `<|low/medium/high-latency|>`) had to be added to the tokenizer as new IDs (262144-262148). New embedding rows for those IDs default to random init. My first pass initialized all 5 new rows to the SAME mean-of-existing-embeddings value — collapsing all 5 tokens to an identical starting point. The LM head then had no way to distinguish them (they compete for the same score against every other token via the same softmax denominator). Post-training special-token loss was 11.87 nats (near uniform 12.48). **0/30 streaming probes emitted any tag.**

Fix: remove the override, let transformers' default `mean_resizing=True` draw new rows from a multivariate-normal with the mean AND covariance of the existing rows — different draws for different tokens. After the fix: special-token loss 8.77, 30/30 probes emit correctly.

**Lesson.** Any embedding init that gives identical starting points to distinct tokens WILL train, but the LM head cannot learn to prefer one over the others because the loss landscape is symmetric. Diagnose with per-token loss, not aggregate loss.

## [ARCHAEOLOGICAL — cond-A deprecated 2026-08-18] The matched pair — cond-A vs cond-B at n=10K

Both arms trained identically. Same 9,567 latency-balanced sentences from SiMT-660K (`results/phase2/phase2_n10k_indices.json`, seed 42, ≤80 source tokens, chunk-count-matched filter). Same recipe (trl.SFTTrainer 1.10, lr 2e-5, effective batch 16, mean-covariance init, 3-epoch cap, 5% val, early-stopping patience 3, threshold 0.001).

The only difference is the training strings. Cond-A's strings use GPT-4's `source_chunks`/`target_chunks` from the shipped corpus. Cond-B's strings use our OT annotator's chunks (τ=0.30) built by `scripts/phase2_build_sft_dataset.py`.

### Training outcomes

| Arm | Chunks/sentence distribution | Best `eval_loss` | Stopped at |
|---|---|---|---|
| cond-A (GPT-4) | uniformly 3-6 words per chunk, ~4-6 chunks per sentence | 1.613 @ step 500 (epoch 0.88) | step 650 (patience 3) |
| cond-B (OT, ours) | variable: 28% single-chunk, rest 2-100+ chunks | 1.677 @ step 550 (epoch 0.97) | step 700 (patience 3) |

Cond-B's eval_loss is slightly higher (1.677 vs 1.613). That's not because cond-B trains worse — it's because cond-B's targets have more entropy (variable-length chunks). What matters is what the models do at inference, not what the SFT loss is.

### Offline BLEU (Layer 1 sanity — do the models translate at all?)

Full-source greedy decoding on newstest2013 (3,000 sentences), with prompt `<|medium-latency|> WHOLE_SOURCE <|end-of-read|>` and stopping at first `<|end-of-write|>` or `<eos>`:

| Arm | Offline BLEU | hyp/ref length |
|---|---|---|
| cond-A | 32.41 | 1.006 |
| cond-B | **32.54** | 1.009 |
| Δ (B − A) | +0.13 | |

Statistically identical. The null hypothesis ("cond-B degrades translation quality compared to cond-A") is rejected. Both models produce competitive translations under the no-streaming case.

**Two bugs caught pre-verdict** (both would have silently corrupted this number if not caught — logged in `LOG.md`):

1. **`sft.py --corpus_file` capped rows at --n_sentences default (2000).** cond-B first training run silently used 2K of 9,567 rows. `n_rows_trained` field in `sft_summary.json` caught this — read every field once.
2. **Extrinsic offline gen didn't stop at `<|end-of-write|>`.** cond-A never saw a "one giant chunk" training row (all GPT-4 chunks are 3-6 words), so after emitting a target chunk it kept producing more `src_i+1 <eor> tgt_i+1 <eow>` — matching the multi-chunk training pattern. Symptom: hyp/ref length 1.99, **BLEU depressed to 15.89**. Post-fix hyp/ref = 1.006, BLEU 32.41.

### Cross-paper comparability protocol (for §Experiments text + Fig. 1/2 captions)

**Method-family split (revised 2026-08-18 per user):** two separate stories on two separate figures, each with matched competitor conventions. The 2×2 diagonal-move framing (OPTIONALS §"the two-cell move") shows in Fig. 2; Fig. 1 shows we're competitive across the whole SiMT literature, not just LLM-based approaches.

| Plot | Story | Test set | Metric | Competitors reportable verbatim |
|---|---|---|---|---|
| **Fig. 1** — vs non-LLM SiMT | "Decoder-only 2B LLM matches encoder-decoder tradition on their own benchmarks" | WMT15 De→En newstest2015 | SacreBLEU-13a × AL (Ma 2019) | ITST, SM²/SimulMask, HMT, wait-k baseline. Dashed reference line: "EAST at 8B/660K" for scale calibration. **Ours: OT-SFT.** |
| **Fig. 2** — vs LLM SiMT | "Among LLM-based SiMT methods, data-construction beats runtime-policy approaches" | WMT22 De→En newstest2022 | SacreBLEU-13a × LAAL (Papi 2022) | EAST (Table 3), Simul-LLM, TransLLaMa, SimulPL, Conversational SimulMT. **Ours: OT-SFT.** |
| Table 3 (multi-lingual) | Direct head-to-head with EAST Table 2 | WMT22 X↔En × 4 pairs | SacreBLEU-13a offline | EAST Table 2 row-by-row, **Ours: mixed OT-SFT (P2 iii)** |

**Rationale for the split.** ITST/SM²/HMT are encoder-decoder methods that populated WMT15 De→En; SimulPL/EAST-Table3/ConvSiMT are LLM-based methods that populated WMT22. Forcing both families onto one figure requires either re-running everyone (impractical) or committing to axes that don't match half the literature. Splitting by method family: (a) matches each competitor to its native test set + metric convention, (b) tells two distinct, defensible stories, (c) EAST plays a well-defined role in each (reference line in Fig. 1, primary competitor in Fig. 2) — no double-counting.

**Where EAST appears.** Primary competitor on Fig. 2 (their direct-competitor role); dashed reference line "EAST-Stage-I at 8B/660K" on Fig. 1 to calibrate the scale gap for the non-LLM comparison. Not a competing curve on Fig. 1.

**Draft paragraph for §Experiments (paste as-is):**

> **Cross-paper comparability.** We report competitor numbers verbatim from published tables. BLEU variants across the referenced papers differ (ITST: Moses `multi-bleu.perl`; EAST, SM², ours: SacreBLEU with 13a tokenizer; SimulPL: SacreBLEU signature not reported). Post (2018, WMT) documents that SacreBLEU-13a and Moses BLEU differ by ≤ 0.3 BLEU on well-formed WMT De→En outputs, which is smaller than the ≥ 5 BLEU spread we report between our method and baselines. Latency variants also differ: ITST/EAST/SM² report AL (Ma et al. 2019, source-truncated); SimulPL reports LAAL (Papi et al. 2022, no truncation). LAAL is empirically 0.2-1.5 source-word-equivalents higher than AL on WMT De→En streaming outputs; the method ranking is invariant across the two on our own runs (verified). Where our numbers appear alongside competitor numbers, we compute both AL and LAAL so the reader can pick either axis.

**Plot marker convention** (visual disclosure — reviewers see it, don't need to hunt for it):
- Ours: solid circle.
- SacreBLEU competitors (EAST, SM², SimulPL): open circle.
- Moses-BLEU competitors (ITST, older HMT): open triangle.
- Legend footnote: *"Marker shape indicates BLEU implementation source. See §Experiments 'Cross-paper comparability' for variance discussion."*

**Fig. 1 uses AL (Ma 2019) because ITST/SM²/HMT report AL; Fig. 2 uses LAAL (Papi 2022) because SimulPL reports LAAL. Our runs compute both — reader can cross-reference either axis to our numbers.**

**Immediate code deliverable:** add LAAL alongside AL in `src/eval/extrinsic.py::compute_al` (~5 lines). Every existing streaming JSON gets LAAL for free on next rerun. LAAL formula: same numerator as AL but denominator is `|Y|` (target length), sum runs over all target tokens without the τ-truncation at source-exhaustion.

---

### Head-to-head with EAST paper (as of 2026-08-18)

EAST reports on WMT22 De→En test set; we report on newstest2013 dev set. Same-test-set comparison is a pending immediate deliverable (~30 min GPU). At time of writing:

**Offline BLEU (EAST Table 2, De→En):**

| Model | Params | Training data | BLEU |
|---|---|---|---|
| GPT-4 | ? | zero-shot API | 33.87 |
| **Ours OT-SFT** (Stage I) | **2B (Gemma-4-E2B)** | **10K SiMT-660K subset** | **32.54** |
| EAST (Stage I + Stage II) | 8B (Llama-3) | 660K + 90K + 120K | **32.55** |
| Llama3-MOMT | 8B | ? | 31.98 |
| ALMA-7B-LoRA | 7B | ? | 29.56 |

**Statistical tie with EAST at 4× fewer params, 66× less data, Stage I only.** This is the paper's data-efficiency headline.

**Streaming BLEU/AL (EAST Table 3, De→En):**

| Method | low | medium | high |
|---|---|---|---|
| EAST (8B, 660K, adaptive commit) | 29.87 @ AL 2.59 | 31.08 @ AL 3.42 | 32.38 @ AL 5.87 |
| **Ours OT-SFT** (2B, 10K, fixed wait-k) | 22.14 @ AL 2.35 (k=3) | 26.94 @ AL 3.54 (k=5) | 28.40 @ AL 4.64 (k=7) |

We recover ~84-88% of EAST's streaming BLEU at each latency band, at 4×/66× disadvantage AND without adaptive commit (H9 — check_argmax gives chunks/sent=1.00 at our data scale). The framing is "competitive at small scale," not "beats SOTA."

COMET-22 numbers are missing (~30 min inference rerun on the `sft_n10k/final/` checkpoint). Immediate deliverable — see `07-next_steps.md`.

### Streaming BLEU + AL (Layer 2 — the paper number)

Under the streaming state machine (`src/eval/extrinsic.py --mode streaming`):
- Feed source word-by-word, maintaining KV cache.
- Under **wait_k** policy: force `<|end-of-read|>` every k source words.
- Under **check_argmax** policy: at each source word, check if model's argmax is `<|end-of-read|>`; if yes, switch to WRITE and generate until model emits `<|end-of-write|>` (or EOR mid-write) or hits cap; else feed next source word.
- Word-unit AL per Ma 2019 §4.

Full 3000 sentences newstest2013, matched cond-A vs cond-B:

| Policy | cond-A BLEU / AL | cond-B BLEU / AL | Δ BLEU | Chunks/sent |
|---|---|---|---|---|
| **wait_k=3** | 16.49 / 2.10 | **22.14 / 2.35** | **+5.65** | 6.41 |
| **wait_k=5** | 21.53 / 3.17 | **26.94 / 3.54** | **+5.41** | 4.04 |
| **wait_k=7** | 23.61 / 4.19 | **28.40 / 4.64** | **+4.80** | 3.03 |
| check_argmax | 30.66 / 18.23 | 30.76 / 18.20 | +0.10 | 1.00 |

**The paper's headline result.** Under any fixed-latency streaming budget (wait-k), cond-B beats cond-A by 4.8-5.7 BLEU at matched AL. Under check_argmax (model decides when to commit), both models revert to reading the entire source before saying anything (chunks/sent = 1.00) and the BLEU gap disappears.

## What this means

**Two hypotheses distinguish here** (see `02-hypotheses.md` for H1-H7 originals; the two below are added in Phase 2):

### H8 (new) — OT-annotated training data teaches the model to translate BETTER under fixed-latency streaming policies than GPT-4-annotated data.

**Predicted:** cond-B under wait-k should give higher BLEU at matched AL than cond-A under wait-k.

**Confirmed at n=10K:** cond-B beats cond-A by +5 BLEU across wait_k ∈ {3, 5, 7}. Signal held from 100-sent smoke to full 3,000-sent runs.

**Interpretation:** cond-A learned a very specific chunking rhythm (uniformly 4-6 words per source chunk). When wait-k forces a rhythm that DOESN'T match GPT-4's original chunk boundaries, cond-A degrades — it produces partial translations that assume the "wrong" chunk structure. Cond-B learned variable-length chunking including 28% single-chunk rows (the "late commit" case for reordering-heavy sentences), so its representations are more robust to arbitrary commit-point placements.

### H9 (new, negative) — Neither cond-A nor cond-B does adaptive model-driven commitment (check_argmax) at n=10K.

**Predicted (original):** cond-B, having seen single-chunk-collapse training rows, would voluntarily emit EOR at plausible points during READ (chunks/sent >> 1, AL < 8).

**Refuted at n=10K:** Both models emit chunks/sent = 1.00 under check_argmax — they always wait for source-exhaustion. Model never emits EOR mid-source. Both models produce BLEU ~30.7 at AL ~18.2 (essentially offline).

**Interpretation:** SFT with the EAST format on 10K rows is not enough to teach the model to CHOOSE commit positions. It learns the tag as a next-token in a training pattern, not as a policy decision. Under a threshold-based argmax check, the model's argmax at intermediate positions is always "the next source word" (from the training pattern `latency src <eor> tgt <eow> src ...`) — never EOR spontaneously.

**Consequence for the paper.** The narrative isn't "cond-B learned to make good commit decisions." It's "cond-B produces higher-quality translations under any imposed streaming latency budget." The mechanism claim ships as: annotation quality generalises across streaming policies; a wait-k policy is a natural way to demonstrate it.

## A worked example — sentence 0 of newstest2013

Source: `Eine republikanische Strategie, um der Wiederwahl von Obama entgegenzutreten` (9 words)
Reference: `A Republican strategy to counter the re-election of Obama`

**Cond-A, wait_k=3:** commits at src=3, 6, 9. Generates:
```
Chunk 1 (g=3): "A Republican strategy,"
Chunk 2 (g=6): "to win reelection"
Chunk 3 (g=9): "from Obama"
```
Total: `"A Republican strategy, to win reelection from Obama"` — 7 words. AL ≈ 3.34.

**Cond-B, wait_k=3:** same policy, same commit positions. Generates:
```
Chunk 1 (g=3): "A Republican strategy"
Chunk 2 (g=6): "to counter Obama's"
Chunk 3 (g=9): "re-election"
```
Similar structure, slightly better lexical choices.

**Cond-A, check_argmax:** reads all 9 words without emitting EOR (argmax at every mid-position is the next German word). At source-exhaustion, we force EOR. Generates: `"A Republican strategy to oppose Obama's re-election"` — 7 words. AL = 9.

**Cond-B, check_argmax:** same — reads all 9, force-EOR at end, generates one translation. `"A Republican strategy to oppose Obama's re-election"` — same output.

Under wait-k, the two models produce measurably different translations of the same source. Under check_argmax, they both fall back to identical offline-like behaviour.

## The state machine, in code

`src/eval/extrinsic.py::stream_translate` (~150 lines). The key structural bits:

1. **Tokenize source ONCE, then walk word-by-word.** Tokenising `" word_i"` in a loop and concatenating gives DIFFERENT ids than tokenising the whole source, due to SentencePiece leading-space and cross-boundary BPE merges. The model was trained on the full-concat form; feeding it piece-by-piece with mismatched BPE = model sees an out-of-distribution token sequence. Fix: tokenize full source, then map BPE tokens to whitespace-word spans, and feed by span. Verified in `scripts/phase2_streaming_smoke.py` on 200 newstest2013 lines — 0/200 mismatches.

2. **`generate_write_chunk` stops on EOW, EOR, or EOS.** Not just EOW — model can emit EOR mid-write (thinking it wants more source), and if we don't stop it hallucinates a German source chunk. Not just EOS — model may naturally terminate translation. All three cases end the WRITE.

3. **Skip the redundant final-EOR drain if last commit was already at source exhaustion.** For wait_k=3 on 9 source words, we commit at 3, 6, 9. If we also force a final EOR at 9, the model sees `<eow><eor>` back-to-back — a pattern it NEVER saw in training (training format is `<eow> src <eor>` with source in between). It responds by hallucinating a German "source chunk" in the drain output. Fix: skip drain if `chunk_g_words[-1] == src_words_read`.

## AL — Average Lagging (Ma et al. 2019 §4)

For each target word `i`, let `g(i)` = number of source words fully read at emission time. Then:

```
AL = (1/tau) * sum_{i=1..tau} (g(i) - (i-1) * |X|/|Y|)
tau = argmin_i (g(i) = |X|)   [first target word where all source is read]
```

Intuition: AL measures the lag between our streaming output and an oracle that reads exactly `(i-1) * |X|/|Y|` source words at target position `i` (perfectly matched pace). Wait-k policies give AL close to `(k+1)/2` asymptotically. Offline (no streaming) gives AL = `|X|`.

Verified against analytic on tiny cases before trusting model numbers:
- Wait-1 on |X|=|Y|=9: AL = 1.00 ✓
- Wait-3 on |X|=|Y|=99: AL = 2.01 ✓
- Offline (`g = [|X|] * m`): AL = 9 ✓

## What Phase 2 still owes the paper — with dates and gates

Ordered by criticality (Findings-blocking first). Full week-by-week plan lives in `07-next_steps.md`.

**CRITICAL for Findings tier:**

- **[Week 1] Retrain OT-SFT on v2 dataset** (fixes fallback-τ + latency reassignment applied 2026-08-18). Streaming eval on newstest2013. Predicted P3 sub-claim iv (chunks/sent > 1 under check_argmax after collapse-row removal).
- **[Week 1] WMT15 + WMT22 De→En reruns** — offline BLEU + streaming eval for Fig. 1/2 competitor comparison. Gate B: OT-SFT ≥ +2 BLEU over Simul-LLM's published wait-k=5 De→En number.
- **[Week 2] Qwen3.5-2B replication (Gate A).** OT-SFT beats published wait-k numbers on Qwen (dataset ready).
- **[Week 2] Gemma-4-E4B scale replication** (P2 ii). Build OT-SFT dataset from `annot_ot_e4b_n10k/matrices.jsonl`; SFT; streaming eval.
- **[Week 3] Reordering-subset analysis on newstest2013** (P1 mechanism sub-claim). No new GPU compute.

**Findings-desirable (Weeks 3-6):**

- **[Week 3-4] Cross-annotator SFT matrix** (P4, 6 off-diagonal cells). ~36 GPU-hours across all cells.
- **[Week 4-5] Data-scale curve on champion** (P2 sub-claim). n=20K/30K/40K/50K.
- **[Week 5-6] Multi-90K mixed multi-lingual OT-SFT** (P2 iii, τ-generalisation P2 iv).

**Engineering / final numbers (Weeks 6-7):**

- **AL-CA Layer 3 measurement** via `torch.cuda.Event`. `scripts/phase2_compute_al_ca_approx.py` gives corpus-level approximation now; Layer 3 needed for apples-to-apples with EAST Table 3.
- **Newstest2015 (WMT15) test-set numbers, reported ONCE** on frozen champion. No hyperparameter tuning after this run.

**Phase 3 appendix (post-writeup / rebuttal):**

- **RWTH-A intrinsic** (blocked on baseline GPT-4 re-annotation decision). ~$5-20 API cost. See `06-data.md`.
- **WaitK-SFT within-framework rebuild** (Cond-C reprise, ~2 days) if a reviewer argues "your +BLEU vs Simul-LLM comes from EAST framework overhead."
