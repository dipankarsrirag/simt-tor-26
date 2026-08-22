# The annotator, mechanically

Read `../METHOD.md` first for the formalism. This document is the implementation-shaped version — what the code actually does, what shape each object has, and why.

**Currency note (2026-08-22):** The annotator itself is unchanged since Phase 1. The v6b pipeline (2026-08-22) changed the DOWNSTREAM steps (SFT recipe + streaming inference) but not the annotator. One post-processing step was added on the SFT-dataset side: EAST §3.1 chunk merging (`scripts/phase2_build_sft_dataset.py --merge_small_chunks --min_src_words 4`) folds chunks with < 4 source words forward into the next chunk. This runs AFTER `_chunks_from_commit`; the annotator's commit points and word-boundary snapping are unmodified.

## The one-sentence version

Given a parallel pair `(source, target)`, decide **per target token** the earliest source-prefix length at which the model's next-token distribution has converged to what it would predict with the full source. Interleave the resulting chunks into EAST's training format.

## The seven-step pipeline

Every call to `annotate_pair(model, tokenizer, source, target, ...)`:

**1. Tokenise source and target independently.**
- `src_ids = tokenizer(source, add_special_tokens=False)["input_ids"]` — length `n`.
- `tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]` — length `m`.
- Source/target token boundaries are what commits are indexed by (`i` for source, `j` for target).

**2. Build the full-source forward pass and extract `P_full[j]`.**
- Input = `bos + prompt(source_full_string) + target_tokens`.
- One forward pass → logits of shape `(1, L, V)` where `V` = vocab size.
- Softmax the logits at the position that predicts each target token → `P_full` of shape `(m, V)`.

**3. Sweep source prefix lengths and extract `P_pre[i][j]`.**
- For `i = 1, ..., n`:
  - `source_prefix_str = tokenizer.decode(src_ids[:i])`.
  - Input = `bos + prompt(source_prefix_str) + target_tokens`.
  - Forward pass → softmax → `P_pre[i]` of shape `(m, V)`.
- Not batched yet; `n` forward passes per sentence. ~1.3s per sentence on H200 for typical 30–50 token pairs.

**4. Compute divergence `D(P_full[j], P_pre[i][j])`.**
- Registered in `CRITERIA` dict:
  - `js` — Jensen–Shannon divergence, symmetric, bounded `[0, ln 2] ≈ [0, 0.693]`. Cheap.
  - `kl` — asymmetric `KL(P_full || P_pre)`. Cheap.
  - `ot` — embedding-grounded optimal transport (METHOD §3 primary). Top-`k` support union, cost `1 - cos(E_a, E_b)` from input embeddings `E`, log-stabilised Sinkhorn via `pot`'s `ot.bregman.sinkhorn_log`. `k=128`, `eps=0.05`, 200 iterations.

**5. Determine commit points.**
- `i*[j] = min { i : D(P_full[j], P_pre[i][j]) < tau }`.
- Fallback: `i*[j] = n` if criterion never fires.
- Then enforce monotonicity: `i*[j] := max(i*[j], i*[j-1])`. A read/write policy cannot un-read source.

**6. Group into chunks.**
- Run-length-encode `i*[]`. Consecutive target tokens sharing an `i*` value become one write chunk.
- Each chunk pairs with the read span from the last commit up to this one.
- If the tail of the source is unread after the last commit, glue it onto the final source chunk (the model must have read everything by end-of-sentence).

**7. Emit EAST format.**
- `<|low-latency|>` / `<|medium-latency|>` / `<|high-latency|>` prefix (from the shipped `latency` field for now; will be `tau`-derived later).
- Interleave: `latency + [src_chunk_k, <|end-of-read|>, tgt_chunk_k, <|end-of-write|>] × K`.
- This string is what the SFT trainer will consume.

## Prompt template — matters more than you'd think

The annotator wraps the source in a **prompt template** before the forward pass. Options currently supported:

- **`raw`** — `f"{source_prefix}\n{target}"`. Six tokens of overhead. Matches METHOD §1's spec — pure next-token prediction.
- **`chat`** — Gemma's chat template with an explicit instruction:
  ```
  <bos><|turn>user
  Translate the following German text to English.

  German: {source_prefix}

  English:<turn|>
  <|turn>model
  {target}
  ```
  28 tokens of overhead.

**When to use which:** raw for base pretrained models; chat for instruction-tuned models. Instruction-tuned models under raw treat the input as document continuation, not translation — the P_pre → P_full convergence then tracks "how much source-language text has accumulated" rather than "how much translation-relevant context." This confound distorted our first Phase-1 sweep; see `hypotheses.md` H1–H3 for the diagnosis.

## Divergence values in the matrix — what shape are they?

`annotate_pair(..., return_full_matrix=True)` populates `AnnotatedPair.divergence_matrix` as a `(n, m)` list of lists — row `i-1` holds `D(P_full[j], P_pre[i][j])` for `j = 0..m-1`. Row `n-1` (full source) is trivially near zero because `P_pre[n] == P_full`.

`return_full_matrix=True` costs nothing extra when we're doing the forward passes anyway. It enables **offline tau sweeps** — try many `tau` values from one GPU run. Also lets us plug new criteria in offline provided the shape is right.

`record_entropy=True` additionally populates `entropy_matrix` `(n, m)` = `H(P_pre[i][j])` in nats, and `entropy_full` `(m,)` = `H(P_full[j])`. Used for the entropy-only ablation (`hypotheses.md` H4).

## Sanity checks the annotator does not enforce (yet)

The METHOD §8 checks (positional-degeneracy Pearson, non-trivial trace shape, comparison against GPT-4 tags) are computed in the sweep-report scripts, not inside `annotate_pair`. See `experiments.md` for which scripts run which check.

## Compute cost

- One sentence, JS/KL, 40 source tokens × 30 target tokens: ~1.3 s on H200.
- OT is ~5–10× JS per pair (Sinkhorn iterations on a 256×256 cost matrix). ~5–15 min for 48 sentences at n≈40 on H200.
- Full 660K corpus at ~1.3 s/sentence with JS would be ~10 GPU-days at stride=1. **We annotate 10–50K, not 660K.** Justified by EAST Figure 6 which shows 10K already gets most of the benefit.
