# Log

Append-only. Newest at the top. Two kinds of entry: **decisions** (what we chose and why) and **runs** (what we executed and what happened).

Log the run *before* starting the next one. A run without an entry did not happen.

---

## Template — decision

```
### [DECISION] YYYY-MM-DD — one-line summary
**Context:** what prompted this
**Options:** what was considered
**Chose:** what and why
**Revisit if:** the condition that would change this
```

## Template — run

```
### [RUN] YYYY-MM-DD — run-id
**Config:** backbone / data size / criterion / tau / seed
**Command:** exact invocation
**Result:** numbers, with the metric named
**Read:** what this means for the next step
```

---

<!-- entries below -->

### [RUN] 2026-08-17 — newstest2013 De→En fetched as dev set for extrinsic harness
**Config:** sacrebleu-hosted WMT13 news-test set; 3,000 De→En sentence pairs.
**Command:** `sacrebleu -t wmt13 -l de-en --echo src 2>/dev/null > newstest2013.de` (and `--echo ref` for `.en`) at `/g/data/po67/dipankar/data/simt-tor-26/wmt13-de-en/`.
**Result:** 3000/3000 aligned. First De line: "Eine republikanische Strategie, um der Wiederwahl von Obama entgegenzutreten". (First attempt captured sacrebleu's stdout download banner into the .de file — re-fetched with stderr silenced.)
**Read:** Dev set for the extrinsic harness. The pipeline (BLEU + AL + AL-CA under streaming) is validated on newstest2013 before newstest2015 is touched — prevents the reviewer-visible test-set numbers from being reported for a buggy inference loop.

### [RUN] 2026-08-17 — cond-B n=10K OT annotation kickoff (job 176455997 → chained 176459737)
**Config:** Same 9,567 latency-balanced indices as cond-A n=10K (`results/phase2/phase2_n10k_indices.json`, seed 42, max_src_tokens=80). OT criterion (`ot_divergence_row_batched`), τ grid `{0.30, 0.50, 0.70, 1.00}`. Pre-seeded with the 1,894 rows from `annot_ot_condB_n2k/matrices.jsonl`; chained self-resubmitting 1h shards via `jobs/phase2_annot_ot_condB_n10k_shard.pbs`.
**Command:** `qsub jobs/phase2_annot_ot_condB_n10k_shard.pbs`.
**Result (mid-run, shard 1 landed):** 7,246 / 9,567 rows annotated (~76%) at time of log. Batched-OT throughput ~2s/sentence on H200 (was 28s/sentence per-pair — 14× speedup; see companion RUN entry). Shard 2 (176459737) queued in H (afterany-dep) state, will pick up remaining ~2,320 rows on next launch. Verified: no NaNs in matrices.jsonl, `\|end-of-read\|` traces present and non-degenerate on sample walk.
**Read:** On track to complete ~4 hours after shard-1 start. Next: build cond-B n=10K dataset (`phase2_build_condB_dataset.py --tau 0.30`), then cond-B n=10K SFT with the same recipe as cond-A n=10K (early-stopping wired, same 3-epoch cap, lr 2e-5, effective batch 16).

### [DECISION] 2026-08-17 — PBS chain-at-START pattern for self-resubmitting shards
**Context.** Cond-B n=10K first shard (176447xxx) hit its 1h walltime; PBS `SIGKILL`d the wrapper *before* the post-python `qsub` could fire. `shard_counter` stuck at 1, no resubmit happened, human intervention required. Same failure mode would have hit the n=2K shard-based pipeline too but we got lucky and it converged within one shard.
**Options.**
- (a) Larger walltime + trust python to exit cleanly (fragile — no recovery if OT hits a numerical corner case).
- (b) Chain the *next* shard's `qsub` at the very *start* of the wrapper, gated on this job's exit status via `-W depend=afterany:$PBS_JOBID`. PBS holds the successor in H state; on any exit (clean, walltime kill, OOM) it launches. First act of the successor is to check for a `DONE` marker and exit cleanly if annotation is complete — avoiding a runaway resubmit loop. `MAX_SHARDS=10` cap as belt-and-suspenders.
- (c) Move to array jobs. Rejected — indices are stateful (each shard's `--resume` skips what's on disk), array semantics don't match.
**Chose:** (b). Implemented in `jobs/phase2_annot_ot_condB_n10k_shard.pbs` lines 63–80. Verified end-to-end: shard 176455997 fired shard 176459737 within seconds of its own launch, `qstat` shows shard 2 in H state pinned to shard 1's completion.
**Revisit if:** any downstream shard-based pipeline (SFT resume, extrinsic-eval resume) shows the same walltime-kill-loses-resubmit failure. Copy this pattern; do not reinvent.

### [RUN] 2026-08-17 — cond-A n=10K SFT with early stopping — best eval_loss 1.613 @ step 500 (job 176432676)
**Config:** `src/train/sft.py --indices_file results/phase2/phase2_n10k_indices.json --num_epochs 3.0 --per_device_batch_size 4 --grad_accum_steps 4 --learning_rate 2e-5 --warmup_steps 50 --logging_steps 25 --eval_steps 50 --val_frac 0.05 --early_stopping_patience 3 --early_stopping_threshold 0.001 --sample_generations 3 --output_dir results/phase2/sft_condA_n10k`. Gemma-4-E2B base with extended tokenizer, effective batch 16, bf16, trl.SFTTrainer 1.10, `completion_only_loss=False`, mean-covariance embedding init (default; see 2026-08-16 embedding-init fix). 9,567 indices kept after 80-tok + chunk-count filters.
**Result:** Early stopping fired at step 650 (epoch 1.144). Best `eval_loss=1.6130` at checkpoint-500 (epoch 0.881); patience-3 window `1.6144 → 1.6443 → 1.6368` all failed to improve by 0.001, `load_best_model_at_end=True` restored step-500 weights.
- **Eval-loss trajectory:** 2.845 (step 50) → 2.441 (100) → 1.795 (150) → 1.665 (200) → 1.660 (250) → 1.640 (300) → 1.635 (350) → 1.631 (400) → 1.625 (450) → **1.613 (500)** → 1.614 → 1.644 → 1.637.
- **Train wall time:** 1,976s (~33 min) for 650 optimizer steps (~3s/step, effective batch 16 on H200).
- **Special-token embedding movement L2:** EOR 0.077, EOW 0.079, LOW 0.082, MED 0.083, HIGH 0.084. All ~2× the 2K/3e values (0.10–0.15 was for full 3 epochs at n=2K; here 1.14 epochs at n=10K moves less per token but on 5× data).
- **Streaming smoke (job 176452xxx, `scripts/phase2_inference_smoke.py`, 40 probes, seed 142):** **40/40 emit both `<|eor|>` AND `<|eow|>` in correct alternation.** Sample gen for idx=405252 (medium latency, prefix "Für Josephus ist"): `es ein Segen, <|eor|> For Josephus it is a blessing <|eow|> dass er die Möglichkeit hat, <|eor|> that he has the opportunity <|eow|> …`.
**Read:** cond-A n=10K trained faster than a fixed-epoch schedule (early stop at ~1.14 epochs vs 3.0) and generalises: eval loss plateaus at 1.61 while train loss keeps dropping (would overfit). Streaming behaviour is clean. Ready to run the same recipe on cond-B once its annotation completes. Extrinsic harness (BLEU + AL + AL-CA on newstest2013 dev) is the next unblocker.

### [DECISION] 2026-08-17 — Wire early stopping + validation split into src/train/sft.py
**Context.** cond-A n=2K/3e (176402113) trained for a fixed 357 steps without a held-out eval. No mechanism to detect overfitting or converge-and-stop; scaling to 10K/50K under the same schedule would either underfit (3 epochs too few) or overfit and waste compute. Reviewers will ask "how did you pick 3 epochs?".
**Options.**
- (a) Report train-loss trajectory only. Rejected — cannot separate memorisation from generalisation on a corpus this small.
- (b) Add explicit `--val_frac` (default 0.05), `--eval_steps`, `--early_stopping_patience`, `--early_stopping_threshold` flags; wire `EarlyStoppingCallback` and `load_best_model_at_end=True`. Same recipe used for both A and B — apples-to-apples.
- (c) Full 3 fixed epochs on all scales, defend post-hoc. Rejected — cost scales linearly, no principled stop.
**Chose:** (b). Rationale:
1. Standard practice for SFT; reviewers expect it.
2. Matched A-vs-B needs both arms to stop at "converged", not at an arbitrary step. If A converges at 1.14 epochs and B needs 2.5, letting each go to its best `eval_loss` is the fair comparison. Enforcing the same wall-clock or step count would penalise whichever converges slower.
3. Cheap: 5% held-out from the same latency-balanced 9,567; eval every 50 steps adds <5% overhead.
**Implementation:** `src/train/sft.py` gained `--val_frac`, `--eval_steps`, `--early_stopping_patience` (default 3), `--early_stopping_threshold` (default 0.001). Val split is deterministic (seed 42) and excluded from the train indices logged to `train_indices.json`. `EarlyStoppingCallback` from `transformers.callbacks`.
**Revisit if:** eval-loss diverges from BLEU/COMET on extrinsic eval (unlikely at n=10K; possible at n=2K). Fallback would be to eval on newstest2013 dev directly every N steps — more expensive but a truer downstream signal.

### [RUN] 2026-08-17 — cond-B n=2K SFT completed (job 176422xxx) — 3 epochs, no early stopping
**Config:** Same recipe as cond-A/fixed 2K/3e except `--corpus_file results/phase2/condB_n2k_dataset.json` (built from `annot_ot_condB_n2k/matrices.jsonl` at τ=0.30, `collapse_policy=keep`). 1,894 sentences, 3 epochs, batch 16 effective, lr 2e-5, mean-covariance init. Run predates the early-stopping wire — fixed schedule for parity with cond-A/fixed 2K/3e.
**Result:** 357 steps @ 3.0 epochs, no eval split. Loss 4.74 (25) → 1.13 (250) → 1.11 (350). Final checkpoint saved to `results/phase2/sft_condB_n2k/final/`. Note: `sft_summary.json` failed to serialise (PosixPath from new `--corpus_file` not str()'d) — fixed in-repo, model saved OK.
**Read:** cond-B pipeline validated end-to-end on n=2K. This is the "does OT-annotated data train at all" smoke. The A-vs-B comparison at 2K is *not* the paper claim (n too small for a defensible extrinsic delta); the 10K result — pending — is what carries the paper.

### [RUN] 2026-08-16 → 08-17 — Batched OT annotator: 14× speedup, matches per-pair within 7e-6 L∞
**Context.** Per-pair OT (`ot_divergence_pair` → `ot_divergence_row`) at 28s/sentence made cond-B annotation at n=10K a ~78-GPU-hour job. Advisor spec (from previous session): batched log-domain Sinkhorn across all m target positions, one GPU-saturating call per source-prefix length.
**Implementation:** `src/annotator/criterion.py` gained `ot_divergence_row_batched()`. Log-domain updates:
```
log_v[b,j] = log_b[b,j] - logsumexp_i(log_K[b,i,j] + log_u[b,i])
log_u[b,i] = log_a[b,i] - logsumexp_j(log_K[b,i,j] + log_v[b,j])
```
Support handling (the subtle bit): each row's support = topk(p_full) ∪ topk(p_pre). Per-pair impl uses `torch.unique`, giving variable-size supports. Batched impl fixes size to `S = 2*topk` including possible duplicates, then zeros duplicate positions in the probability vectors before renormalising — equivalent semantics under Sinkhorn (duplicates contribute zero mass) without requiring ragged tensors. First-cut kept duplicates in mass-space (extra support for the regulariser to exploit) → L∞ diff 0.033 vs per-pair. Fixed by explicit dedup-by-sorting + gather-back-to-original-order; final L∞ diff 7e-6 on CPU test with (V=500, D=16, m=25, topk=32, eps=0.05, iters=100).
**Command (verification):** `python scripts/phase2_batched_ot_smoke.py`
**Result:**
- CPU smoke: per-pair 4.2s, batched 0.4s (~10× on tiny problem).
- H200 (cond-B n=10K): ~2s/sentence batched vs 28s/sentence per-pair (~14×).
- L∞ diff `7e-06`, L1 `4.8e-05`, L2 `9e-06` on 25 pairs → **PASS** within 5e-3 Sinkhorn tolerance.
**Read:** Pure engineering win, zero semantic drift. `make_ot(batched=True)` default; per-pair path retained (`batched=False`) as reference impl for future correctness checks. cond-B n=10K annotation now compute-feasible in <8h wall (was multi-day). Verified on-corpus by re-annotating the 1,894 n=2K indices with batched impl and checking commit-trace parity against the per-pair run (spot-checked 3 indices, matched to Sinkhorn tolerance).

### [RUN] 2026-08-16 — Phase 2 Gate 2 PASSES on cond-A after embedding-init bug fix + hand-off state
**Summary:** Extended tokenizer + trl.SFTTrainer wrapper + full cond-A 2K×3-epoch training + verify/smoke jobs all landed; cond-B OT annotation set up as self-resubmitting 2h shards and left running for the overnight.

**Bug diagnosed and fixed (load-bearing, would have poisoned every A-vs-B comparison).**
Initial cond-A SFT on 2K/1e (job 176399546, walltime 8:31) had loss going 4.45 → 2.72 and embeddings moving 0.02–0.04 L2 — but 0/30 streaming probes emitted `<|eor|>`/`<|eow|>`. Diagnostic `scripts/phase2_verify_loss.py` (job 176401727) showed special-token loss median 11.81 nats vs content-token median 0.94 — model was essentially uniform-random on special tokens; top-1 at 0/11 special positions.
Root cause: `src/train/sft.py` overrode transformers's mean-covariance embedding init with a plain `in_emb[orig_vocab:] = mean_in`, collapsing all 5 EAST tokens to the identical starting point. Removed the override; the transformers default (multivariate-normal with old rows' mean and covariance) gives distinct random starts. Fix committed as part of this session.

**Fixed cond-A 2K×3e (job 176402113, walltime ~23min).**
- Loss: 4.45 → 2.72 over first 119 steps (same as buggy), continues → 2.10 by step 357.
- Special-token embedding movement L2: 0.10–0.15 (vs 0.02–0.04 buggy).
- Special-token loss (verify job 176406443): mean 8.77, median 9.14 (vs 11.87 / 11.81 buggy). Top-1 at 10/11 special positions correct (vs 0/11 buggy). Only pos 0 (predict `<|low-latency|>` after BOS) still wrong — not a generation problem because we always feed the latency token in the prompt.
- Content-token loss (verify): mean 1.25 / median 0.21 (vs 2.24 / 0.94).
- Streaming smoke (job 176406444, `scripts/phase2_inference_smoke.py`, 30 probes at 3-word prefix + latency): **30/30 emitted both `<|eor|>` AND `<|eow|>`, all in correct EOR-before-EOW alternation, median EOR position 8 tokens into generation.**

**Gate 2 verdict: PASSES.** Training pipeline validated end-to-end. Ready for cond-B on matched indices, then A-vs-B extrinsic.

**Cond-B OT annotation kicked off (job 176408506, first shard).**
20h monolithic job (176400901) killed in favour of self-resubmitting 2h shards via `jobs/phase2_annot_ot_condB_n2k_shard.pbs`. Same 1894 indices as cond-A (`results/phase2/phase2_n2k_indices.json`), OT criterion, extended τ grid `{0.30, 0.50, 0.70, 1.00}`. `phase1_tau_sweep.py --resume`: reads existing matrices.jsonl on start, skips processed indices, appends new rows with per-row flush+fsync (mid-sentence kill loses ≤1 row). Writes DONE marker when all indices in; NEEDS_RESUME otherwise, triggering the wrapper to `qsub` itself again. Cap `MAX_SHARDS=15`. Expected: ~260 sentences per 2h shard, ~8 shards total for 1894 indices.

**Files landed this session.**
- Scripts: `phase2_prepare_tokenizer.py`, `phase2_inference_smoke.py`, `phase2_verify_loss.py`, `phase2_build_condB_dataset.py`.
- Infrastructure: `src/train/{__init__,sft.py}`, `results/phase2/tokenizer-extended/` (versioned 5-EAST-tokens tokenizer at ids 262144–262148), `results/phase2/phase2_n2k_indices.json` (deterministic 1894-index sample).
- Jobs: `phase2_{toy_sft, sft_condA_n2k, sft_condA_n2k_e5, sft_condA_n2k_fixed, verify_loss, verify_loss_fixed, smoke_condA_n2k, smoke_condA_fixed, annot_ot_condB_n2k, annot_ot_condB_n2k_shard}.pbs`.
- Results (committed): `sft_condA_n2k_fixed/{sft_summary,train_indices}.json`, `smoke_condA_n2k{,_fixed}.json`.

**Pick-up-tomorrow state.**
1. Check `results/phase2/annot_ot_condB_n2k/{DONE,NEEDS_RESUME,matrices.jsonl}` — if DONE present, all 1894 sentences annotated.
2. Run `python scripts/phase2_build_condB_dataset.py --tau 0.30` → `results/phase2/condB_n2k_dataset.json`.
3. Submit cond-B SFT: same recipe as cond-A/fixed but `--corpus_file results/phase2/condB_n2k_dataset.json --output_dir results/phase2/sft_condB_n2k` (n=2000, 3 epochs, lr 2e-5, effective batch 16).
4. Run inference smoke on cond-B; matched A-vs-B qualitative comparison.
5. Scaffold `src/eval/extrinsic.py` for Gate-3 (streaming inference + BLEU + AL on WMT15 newstest2015).

### [RUN] 2026-08-16 — Phase 2 toy SFT job 176399349 — completed
**Config:** `src/train/sft.py` with `--n_sentences 100 --max_steps 20 --per_device_batch_size 2 --grad_accum_steps 2 --warmup_steps 2 --logging_steps 1 --sample_generations 3`. Gemma-4-E2B base with extended tokenizer (`results/phase2/tokenizer-extended/`, vocab 262,149). Condition A (shipped GPT-4 chunks). bf16, trl.SFTTrainer 1.10, `completion_only_loss=False`. Walltime 00:05:45 (cput 00:09:10). One H200.
**Command:** `qsub jobs/phase2_toy_sft.pbs`
**Result:** Exit 0. Kept 95/100 sentences after 80-tok filter. Loss 4.39 → 2.73 over 20 steps (noisy, expected at this sample size / step count). Mean token accuracy 0.51 → 0.59. Model saved to `results/phase2/toy_sft/final/`.
- **Special-token embedding movement (L2 Δ over 20 steps):** `<|end-of-read|>` 0.0035, `<|end-of-write|>` 0.0037, `<|low-latency|>` 0.0024, `<|medium-latency|>` 0.0022, `<|high-latency|>` 0.0020. All nonzero → not loss-masked out.
- **Post-train greedy generations (3 samples):** none emitted `<|eor|>`/`<|eow|>`. Expected — 20 steps on 100 samples is smoke, not real training. Also my generation prompt fed the whole source instead of streaming a prefix (needs fix in the extrinsic-eval harness — noted for Gate 3, not blocking Gate 2).
- **Verification post-hoc (`python -c ...`):** training strings for idx=190712 interleave 5 `<|eor|>` + 5 `<|eow|>` + 1 `<|low-latency|>` correctly. Each token tokenizes to a single id (262144-262148). No multi-piece garbage.

**Read.** Toy SFT smoke passes. trl.SFTTrainer + Gemma-4-E2B + extended tokenizer + EAST interleave format run end-to-end without errors. Special tokens are seen by the model and their embeddings train. Ready to scale to condition-A on 2K (Gate 2 proper).

### [DECISION] 2026-08-16 — Phase 2 kickoff sequencing: SFT scaffold first, annotation second
**Context.** Gate 1 landed (OT PASSES, JS FAILS — Phase 2 unblocked per `TIMELINE.md`). Naive "start Phase 2" reading was "submit 10K OT annotation." But OT costs 28s/sentence × 10K = 78h — over the 48h walltime cap, forcing a sharded submission with no validated downstream. Condition A (GPT-4 tags) needs zero annotation — the tags ship with SiMT-660K.
**Options.**
- (a) Submit 10K OT annotation (sharded 5×15h) NOW, scaffold SFT during the wait.
- (b) Scaffold SFT wrapper first (no GPU), validate on shipped condition-A tags via toy SFT (~15 min GPU), then annotate condition-B on a small subset (2K) to close the pipeline end-to-end. Scale to 10K once the 2K loop lands.
**Chose:** (b). Rationale (from advisor):
1. If trl.SFTTrainer has a special-token gotcha and we've already burned 78 GPU-hours on OT annotation, we lose a week.
2. Condition-A SFT is fully compute-decoupled from annotation — should be validated first.
3. 2K matches EAST Fig. 6's smallest data-size point — the 2K → 10K → 50K trajectory is a paper-relevant ablation for free.
4. Sequenced-small-first mirrors the same "start small, then scale" rule that governed Gemma-4-E2B primary selection in the 2026-08-14 backbone-switch decision.

**Concrete sequence:**
1. **Now, no GPU:** `scripts/phase2_prepare_tokenizer.py` — add 5 EAST special tokens to Gemma-4-E2B tokenizer, save to `results/phase2/tokenizer-extended/`. Versioned once; used consistently by SFT and inference (advisor blocker: tokenizer drift between annotate/train/infer breaks every downstream metric).
2. **Now, no GPU:** `src/train/sft.py` — trl.SFTTrainer wrapper. Loads extended tokenizer + resized model. Builds EAST-interleaved strings from `source_chunks`/`target_chunks`. **Full-sequence CE loss (not completion-only)** per EAST §3.2 — see RELATEDWORKS.md §EAST-#3 note that this is an intentional break from Wang et al. 2024.
3. **~15 min GPU:** toy SFT — 100 rows of shipped GPT-4-chunked SiMT-660K, condition A, 20 steps. Verify (i) special-token embeddings move, (ii) trl loop completes, (iii) a post-train generation places `<|end-of-read|>`/`<|end-of-write|>` markers plausibly.
4. **After (3) works:** condition-A SFT on 2K subset (latency-balanced, seed 42, ≤80 tok filter — matches EAST Fig. 6). ~1-2h GPU. **This is Gate 2.**
5. **Parallel to (4):** OT annotation on the SAME 2K indices. Either sharded (5×~3h) or batched (~2h with M10 speedup). Deferred until after (3) — no point burning SU before pipeline is validated.
6. **After (4) and (5):** condition-B SFT on the 2K OT-annotated rows. Matched A-vs-B extrinsic on WMT15 newstest2015 (BLEU/COMET/BLEURT vs AL/LAAL/AL-CA).
7. **Scale gate:** if 2K matched A-vs-B looks defensible, scale annotation + SFT to 10K, then 50K.

**Blockers to preempt (advisor):**
- **Tokenizer consistency.** Annotate/train/infer must use the same extended tokenizer.
- **Loss recipe.** EAST §3.2 computes CE on source + target + special tokens. NOT `DataCollatorForCompletionOnlyLM`.
- **KV-cache preservation at inference.** EAST inherits ~49 ms/word from interleaved-format autoregressive inference. Critical for AL-CA reporting at Gate 3.

**Sample selection.** Do NOT reuse the Gate-1 stratified 210 for training (would make the intrinsic claim circular). Fresh latency-balanced sample, seed 42, ≤80 tok filter, sizes 2K → 10K → 50K matching EAST Fig. 6.

**Revisit if:** toy SFT fails at (3) — diagnose before running (4). Or if the OT annotation walltime remains prohibitive after batching (M10) is done — fall back to sharded submission (5×3h).

### [RUN] 2026-08-16 — precompute GPT-4 Pearson on 660K + stratified sample (login node, no GPU)
**Config:** `scripts/phase1_precompute_gpt4_pearson.py` (batched tokenizer, `BATCH_SIZE=5000`), tokenizer `MODEL_BASE/gemma-4-E2B`, max_src_tokens=80 (matches sweep filter), bin thresholds monotone ≥ 0.90 / reordering < 0.70, n_per_bin=70, seed=42.
**Command:** `python -u scripts/phase1_precompute_gpt4_pearson.py`
**Result:** 660,876 rows processed in 113.9s (≈5,800 rows/s on login node). Kept 631,915 after 80-token filter (28,961 skipped, 0 failed). **Bin distribution: monotone 74.3% (469,332) / mild 24.4% (154,133) / reordering 0.7% (4,272) / undefined 0.7% (4,178).** Stratified-sampled 210 indices (70 per bin) → `results/gate1/gate1_indices.json`. Full per-sentence table → `results/gate1/gpt4_pearson_full.json`.
**Read:** Reordering bin is genuinely rare (0.7%) because EAST's App. C monotonicity filter already dropped many of the worst reordering cases at data-release time — consistent with the paper's own admission that non-monotonic pairs are excluded. Enough remain (4,272) to sample a defensible 70. Threshold approximation caveat (chunk-independent tokenisation ~1-2 tok slop per chunk) documented in the JSON config.

### [RUN] 2026-08-16 — Gate 1 landed: OT PASSES, JS FAILS. Phase 2 unblocked.
**Jobs:** OT `176387597.gadi-pbs` (cput 01:38:50, walltime 01:38:53, Exit 0). JS `176387598.gadi-pbs` (cput 00:04:42, walltime 00:06:16, Exit 0). Full report: `results/gate1/gate1_report.md`.

**Analysis command:**
- `python scripts/phase1_reordering_bin.py --matrices results/phase1_tau_sweep_ot_n200/matrices.jsonl --gpt4_pearson_full results/gate1/gpt4_pearson_full.json --tau_grid 0.30,0.40,0.50,0.60,0.70,0.80,0.90 --output results/gate1/reordering_bin_ot_n200.json`
- `python scripts/phase1_reordering_bin.py --matrices results/phase1_tau_sweep_js_n200/matrices.jsonl --gpt4_pearson_full results/gate1/gpt4_pearson_full.json --tau_grid 0.02,0.05,0.08,0.10,0.15,0.20,0.30 --output results/gate1/reordering_bin_js_n200.json`

**Result (effective MATCH% = the honest metric — single-chunk collapse counts as MISS):**

| Criterion | monotone | mild | reordering | Verdict |
|---|---|---|---|---|
| OT (winning) | 38.6% (cov 100%) | 60.0% (cov 77%) | **54.3% (cov 77%)** | PASS |
| JS (ablation) | 55.7% (cov 80%) | 44.3% (cov 49%) | 44.3% (cov 46%) | FAIL |

- **OT PASSES both Gate-1 criteria.** Reordering-bin effective MATCH (54.3%) strictly beats monotone-bin (38.6%) by 15.7 pp — mechanism claim ("margin widens on word-order-divergent pairs") confirmed at n=210 stratified. Coverage 77% above the 70% threshold in TIMELINE Gate 1. Monotone-bin chunk-count Δ = 0.67 (tight, ours 4.66 vs GPT-4 4.59). METHOD §8 sanity checks pass (positional Pearson median 0.78 across all bins; zero identity-like traces; 5.7% terminal-degenerate — non-degenerate criterion).
- **Bin-ordering caveat.** Actual bin ordering is `monotone ≪ {reordering ≈ mild}` (38.6 < 54.3 < 60.0), not the strictly-widening `monotone < mild < reordering` the `CLAUDE.md` claim predicts. Conditional MATCH is nearly-monotone (mono 38.6 < reord 70.4 ≈ mild 77.8); the mild-vs-reordering effective gap opens because 16% of the reordering bin remains single-chunk-collapse even under OT (the true late-commit-required tail — see 2 walked examples in `results/gate1/gate1_report.md` §Walked reordering-bin examples). Paper framing should be "bimodal-vs-monotone", not "monotonically widening margin".
- **JS FAILS as a headline criterion.** No mechanism concentration — effective MATCH tied across bins. Root cause is coverage: JS collapses to single-chunk on 54% of reordering / 51% of mild sentences at strict tau because JS doesn't fire when P_pre and P_full concentrate on different-but-semantically-similar tokens. JS remains valid as a cheap ablation for demonstrating OT's advantage; not a viable "ship shorter method section" fallback.

**Interim metric refinement (this session).** Two rounds of correction to `phase1_reordering_bin.py`:
- (1) After JS results, added `MATCH_eff` (treats single-chunk collapse as MISS) alongside `MATCH_cond` — because the initial conditional-only metric was misleadingly high on the reordering bin (single-chunk collapses were being dropped rather than counted as MISS). Pass criteria in `TIMELINE.md` Gate 1 updated to reference effective MATCH and coverage floor 70%.
- (2) After OT results, caught a floating-point corner case: when the matched-count τ produced a commit trace with all identical values, per-sentence Pearson denominator was mathematically zero but computed to ~1e-16 due to FP roundoff in `sum(xs)/m` — yielding a defined Pearson < 0.85 which counted as MATCH. Fixed by requiring `ours_chunks > 1` explicitly in the match predicate. Affected 5 OT-reord + 10 OT-mild sentences. Corrected MATCH_eff dropped from initial reads of 61.4% (reord) / 74.3% (mild) to 54.3% / 60.0%. Verdict unchanged.

**Unlocks:** Phase 2 SFT per `TIMELINE.md`. Annotate 10K then 50K with the OT winning config; matched-condition SFT (A = GPT-4 tags, B = ours) on Gemma-4-E2B; extrinsic eval on WMT15 newstest2015 with BLEU/COMET/BLEURT vs AL/LAAL/**AL-CA**.

**Reservations (all logged; none blocking).**
- Gate 1 measures agreement-with-GPT-4, not gold-alignment tag quality. RWTH-A eval (EAST App. E.4 mirror) runs in Phase 3.
- 16% of the reordering bin (12/70 sentences) is single-chunk collapse even under OT. Expected tail — the true late-commit reorder cases where even OT waits until end. Worth walking examples during writeup.
- Chunk-length whitespace-slop in the precomputed GPT-4 Pearson (~1-2 tok/chunk, documented in `gpt4_pearson_full.json`'s config) is unlikely to move sentences across the 0.90/0.70 thresholds but should be noted if any reviewer asks about bin boundary sensitivity.

---

### [RUN] 2026-08-16 — Gate 1 sweeps submitted, jobs 176387597 (OT) and 176387598 (JS)
**Config:** Gemma-4-E2B base + raw concat, sampled indices from `results/gate1/gate1_indices.json` (210 sentences, ~70 per reordering bin). Two conditions:
- **OT** (winning per Config D-ext): tau grid `{0.30, 0.50, 0.70, 1.00}`. 2:30 walltime (200 × 31s/sentence + overhead).
- **JS** (cheap ablation): tau grid `{0.02, 0.05, 0.10, 0.15, 0.20, 0.30}`. 0:30 walltime (200 × 1.3s + overhead).
**Command:**
- `python scripts/make_job.py --name phase1_tau_sweep_ot_n200 --queue gpuhopper --ngpus 1 --walltime 02:30:00 --script "python scripts/phase1_tau_sweep.py --criterion ot --taus 0.30,0.50,0.70,1.00 --max_src_tokens 80 --prompt_mode raw --model_path /g/data/po67/dipankar/models/gemma-4-E2B --output_dir results/phase1_tau_sweep_ot_n200 --indices_file results/gate1/gate1_indices.json" --output jobs/phase1_tau_sweep_ot_n200.pbs && qsub jobs/phase1_tau_sweep_ot_n200.pbs`
- `python scripts/make_job.py --name phase1_tau_sweep_js_n200 --queue gpuhopper --ngpus 1 --walltime 00:30:00 --script "python scripts/phase1_tau_sweep.py --criterion js --taus 0.02,0.05,0.10,0.15,0.20,0.30 --max_src_tokens 80 --prompt_mode raw --model_path /g/data/po67/dipankar/models/gemma-4-E2B --output_dir results/phase1_tau_sweep_js_n200 --indices_file results/gate1/gate1_indices.json" --output jobs/phase1_tau_sweep_js_n200.pbs && qsub jobs/phase1_tau_sweep_js_n200.pbs`
**Result:** QUEUED — awaiting run. Both jobs land per `docs/next_steps.md` §1; `scripts/phase1_reordering_bin.py` runs on `matrices.jsonl` outputs and produces the Gate-1 stratified table.
**Read:** Pre-flight dry-run of `phase1_reordering_bin.py` against existing n=48 OT-ext matrices (`results/phase1_tau_sweep_ot_ext/matrices.jsonl`) succeeded: 38 monotone / 10 mild / 0 reordering (as expected — n=48 was balanced-latency, not balanced-reordering). MATCH% (ours_pearson < 0.85) was 54% monotone / 78% mild — pattern consistent with mechanism claim (higher agreement on non-monotonic sentences), waiting on n=210 stratified for the reordering-bin verdict.

### [DECISION] 2026-08-16 — Gate 1 redefined: stratified-by-reordering on 200 SiMT-660K sentences; RWTH-A deferred to Phase 3 appendix

**Context.** Prior Gate 1 (per original `TIMELINE.md`) required scoring both ours' and GPT-4's tags on the RWTH De→En manually aligned corpus under EAST Eq. 4 (`A = (1/T) Σ I[a_i ≤ g_i]`). RWTH data has landed; script not yet written. Writing it was blocked on one open choice: what baseline to compare against, since GPT-4 chunks do not exist for the RWTH sentences (RWTH ≠ WMT15-derived SiMT-660K). Additionally: EAST itself put RWTH in App. E.4, not in the main body — the intrinsic result was supporting evidence, not the headline. Session with the user surfaced that the original gate framing may be doing too much work — it was trying to be both a "greenlight for Phase 2" gate and a "paper-headline intrinsic result", and neither role is well-served by that setup.

**Options.**
- (a) Keep Gate 1 as RWTH-A. Write `src/eval/rwth_intrinsic.py`; decide baseline (fast_align / GPT-4-API / wait-k floor); run. Compute-cheap but adds ~1 week of engineering + 1 open baseline decision, and produces a metric on a dataset that EAST relegated to appendix.
- (b) Redefine Gate 1 as a stratified-by-reordering aggregate on 200 SiMT-660K sentences, using GPT-4's own per-sentence Pearson as the reordering-severity proxy. Report per bin (monotone ≥0.90, mild 0.70–0.90, reordering <0.70): chunk-count delta vs GPT-4, per-sentence Pearson, MATCH rate under threshold 0.85. RWTH-A moves to Phase 3 as the paper's App. E result, mirroring EAST's positioning.
- (c) Skip Gate 1 entirely, take the SU risk on Phase 2.

**Chose:** (b). Rationale:
1. Mirrors EAST's own positioning (headline extrinsic, appendix intrinsic).
2. Directly tests the mechanism claim ("margin widens on reordering pairs") in a way RWTH-A does not — RWTH-A gives a single number, this gives a stratified table.
3. Reuses infrastructure we already have — GPT-4 per-sentence Pearson is already computed in `phase1_gpt4_pearson.py`; no new dependency (no awesome-align, no fast_align).
4. Avoids the RWTH-baseline ambiguity — comparing against GPT-4 on the SAME sentences is unambiguous.
5. Compute-cheap: bumps existing n=48 sweep to n=200 (OT ~2h, JS ~15 min). No additional engineering beyond a bin-analysis script.

**Explicit caveat (must survive into any paper draft):** without gold alignment, agreement-with-GPT-4 is *not* tag quality. Gate 1 is a greenlight for Phase 2, not a paper result. The paper's intrinsic story still requires the RWTH-A eval in Phase 3. This caveat is stated in `TIMELINE.md` Gate 1 and `EXPERIMENTS.md` §Two-evaluations-not-one.

**Bin thresholds (fixed absolute, not sample-dependent quintiles — advisor point):**
- `monotone`: GPT-4 per-sentence Pearson(i/n, j/m) ≥ 0.90
- `mild reordering`: 0.70 ≤ Pearson < 0.90
- `reordering`: Pearson < 0.70

Fixed thresholds mean the bins mean the same thing at n=200, n=509 (Phase 3), and any future re-run. Chosen to align with the n=48 top-8 reordering candidates (which had GPT-4 Pearson 0.693 to 0.863 — mostly in the middle bin, one in the reordering bin).

**Pass criteria (Gate 1):**
- Monotone bin: tie GPT-4 on chunk-count delta and per-sentence Pearson.
- Reordering bin: strictly higher MATCH rate (Pearson < 0.85) than the config would produce if it were degenerate (positional or single-chunk).
- METHOD §8 sanity checks all green on the winning tau.

**Additional advisor-recommended step (adopted):** Precompute GPT-4 per-sentence Pearson on the *full* 660K first (~5 min on login node, pure chunk arithmetic — no GPU), then stratified-sample 200 (~70 per bin). Prevents the reordering bin from being sample-noise-dominated at 200 with a balanced-latency (not balanced-reordering) sample. Alternative would be to keep the balanced-latency sample and report CIs — chose to precompute for cleaner numbers.

**Files modified this decision.** `CLAUDE.md` (empirical-status line + dataset table), `TIMELINE.md` (Gate 1 + Phase 3), `EXPERIMENTS.md` (§Two evaluations), `docs/next_steps.md` (reordered §1 = new Gate 1), `docs/data.md` (RWTH note).

**Revisit if:** Gate 1 fails on the n=200 stratified analysis but the winning config was correct at n=48. Would suggest either the bin thresholds are wrong (too strict on reordering) or that the n=48 result was sample-noise. In either case, log the diagnosis and either loosen the pass criteria or investigate the mechanism.

### [SESSION HANDOFF] 2026-08-15 — end-of-session state (Phase 1 mostly landed)

**Where we ended.** Phase 1 explored four annotator configurations and settled on **base gemma-4-E2B + raw concat + OT with extended τ grid** (Config D-ext) as the winning setup. Seven hypotheses (H1–H7) documented in `docs/hypotheses.md`; H1 rejected, H2 partial, H3 supported (aggregate) with per-sentence caveats, H4 provisional support (need finer sweep), H5 SUPPORTED (OT beats JS on beats-random range and per-sentence GPT-4 correlation), H6/H7 queued.

**Best result so far:** Config D-ext (job 176318744): 100% fire coverage; chunk-count 3.98 vs GPT-4's 4.06 (mean_abs Δ = 0.62); 6/8 top-reordering-candidates catch (best of any config); lowest per-sentence Pearson observed = 0.34 on idx=537446. Per-sentence r(GPT-4, ours) = 0.222 (n=47).

**No active jobs at handoff time.** All 6 GPU sweeps completed. Model gemma-4-E2B base downloaded to `MODEL_BASE/gemma-4-E2B` (9.6 GB). RWTH gold alignments extracted at `data/rwth-de-en/DeEn/` (509 sentence pairs, sha256 `5aea49f44a9da4cf575d2dd303a8e12ebe7ba8b615ede7c28e7f8b0a0eb95793` on `DeEnGoldAlignment.tar.gz`).

**Uncommitted work at handoff:**
- Modified: `CLAUDE.md` (slimmed → points at `docs/`), `LOG.md` (this entry + all Phase-1 run entries), `scripts/download_data.sh` (RWTH manual step encoded), `src/constants.py` (Gemma-4 base primary).
- New: `docs/` (7 files: README, method_overview, hypotheses, experiments, random_floor_and_ot, data, next_steps), `results/phase1_*` (6 sweep-result dirs — JSONL matrices + JSON summaries), `jobs/phase1_*.pbs` + `jobs/download_gemma4_e2b.pbs` (7 new PBS scripts), `scripts/phase0_verify_east_format.py` + `scripts/phase1_*.py` (7 new analysis scripts) + `scripts/smoke_load_gemma4.py`, `src/annotator/{__init__, east_format, criterion, annotate}.py` (annotator library), `tests/test_annotator_cpu_tiny.py`, `.venv-freeze.txt` (post-layering freeze; 217 packages).

**Next-session pick-up in one paragraph.** Read `docs/README.md` → `docs/hypotheses.md` → `docs/experiments.md`. The primary Phase-1 result (RWTH Eq. 4 A-score under Config D-ext vs a baseline) is unblocked but not yet computed — write `src/eval/rwth_intrinsic.py` per `docs/next_steps.md` §1. Open follow-ups: bump sample to ~200 (~30 min OT), cross-backbone Qwen3.5-2B (H6), OT sensitivity ablation on topk/eps. Do NOT scale to Gemma-4-E4B until RWTH result is defensible.

**Files a new person should read in order.** `CLAUDE.md`, then `docs/README.md`, then `docs/hypotheses.md`, then `docs/experiments.md` (has all six config sweep tables side by side), then `docs/random_floor_and_ot.md` (intuition for the two concepts that keep coming up), then `docs/next_steps.md`. `LOG.md` is the primary chronological record; `docs/` is the curated summary.

---

### [DECISION] 2026-08-14 — RWTH gold alignments: URL confirmed, manual fetch step
**Context:** `scripts/download_data.sh` step 5 was a TODO; Gate 1 (intrinsic annotation-quality eval, EAST §E.4) is blocked without the RWTH De→En manual alignments. Confirmed from the EAST PDF (arXiv 2504.09570, page 17–18): dataset is "Gold Alignment for Europarl German-English Dataset" v1.0 at `https://www-i6.informatik.rwth-aachen.de/goldAlignment/`, EAST metric is Eq. 4 — `A = (1/T) sum_i I[a_i <= g_i]`, following Zhang and Feng, 2022.
**Options:** (a) script the download, (b) manual browser step, (c) skip and use a substitute alignment source.
**Chose:** (b). The URL is a registration form: name/organisation/email plus a "non-commercial, no redistribution" licence acceptance. Not scriptable in `download_data.sh`. Encoded the manual instructions in the script (step 5) so the human at execution time has all the context in one place. Target directory `data/rwth-de-en/`. HOUSEKEEPING §3 requires a `docs/data/rwth-de-en.md` note post-fetch with filename, date, and sha256.
**Revisit if:** the RWTH form or licence changes, or if a mirrored copy becomes available under redistributable terms.

### [RUN] 2026-08-15 — phase1_tau_sweep_ot_ext 176318744.gadi-pbs — completed
**Config:** same as prior OT run, τ grid extended to {0.30, 0.50, 0.70, 1.00, 1.30}. Reason: prior OT sweep (τ ≤ 0.50) left 4/8 reordering candidates as single-chunk collapses (OT distance stayed above 0.50 across all i,j on those sentences). ~27 min walltime.
**Result:** 33s/sentence. Full sweep:

| τ | fire% | ours_ch | Pearson med | Pearson min |
|---|---|---|---|---|
| 0.30 | 90% | 4.67 | 0.81 | 0.00 |
| 0.50 | 98% | 9.04 | 0.96 | 0.63 |
| **0.70** | 100% | 6.73 | 0.93 | **0.34** |
| 1.00 | 100% | 1.02 | ~0 | 0.00 |
| 1.30 | 100% | 1.00 | ~0 | 0.00 |

Coverage now complete: τ=0.70 gives 100% fire with Pearson_min=0.34 (lowest per-sentence Pearson observed anywhere). τ ≥ 1.00 collapses to single-chunk (fires at i=1 for all target tokens).

**Per-sentence GPT-4-vs-OT (matched-chunk-count tau, grid {0.30, ..., 1.00}):**
- r(GPT-4, ours) = 0.222, n=47 (vs prior narrow-grid 0.306, n=37 — new sentences with imperfect matches lower r but honest).
- Ours chunks_mean = 3.98 (vs GPT-4's 4.06 — essentially matched).
- Chunk-count delta mean_abs = **0.62** (was 1.42 under narrow grid — dramatic improvement).

**Reordering catches (top-8 lowest GPT-4 Pearson): 6 MATCH, 2 MISS.**
- New matches unlocked by extended grid: idx=359904 (0.751), idx=537446 (**0.340** — lowest anywhere), idx=367208 (0.847).
- Remaining MISS (0.87, 0.87) close to threshold — a threshold of 0.87 would flip both.

**Read.** D-ext is the best configuration yet: 6/8 reordering catches, chunk-count matched to GPT-4, coverage complete. This is what goes into the RWTH Eq. 4 arbitration. Per-sentence r dropped slightly (0.306 → 0.222) — but the "6/8 MATCH" and "chunk-count delta 0.62" are stronger evidence for tag quality than r on a monotonic-dominated dataset.

### [RUN] 2026-08-15 — phase1_tau_sweep_ot 176307323.gadi-pbs — completed
**Config:** backbone gemma-4-E2B (base), same 48 sentences (seed 42, max_src_tokens=80), criterion **OT** (embedding-grounded optimal transport via `pot.bregman.sinkhorn_log`, topk=128, eps=0.05, 200 Sinkhorn iterations), tau grid {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}, prompt raw. Walltime 01:00:00 on 1×H200 gpuhopper.
**Command:** `qsub jobs/phase1_tau_sweep_ot.pbs` (using `pot`'s `ot.bregman.sinkhorn_log` after user request; original hand-rolled Sinkhorn cancelled and replaced).
**Result:** 25 min annotation (~31s/sentence — ~24× slower than JS due to Sinkhorn iterations on ~256×256 cost matrices). Full sweep:

| τ | fire% | ours_ch | Pearson med | Pearson min |
|---|---|---|---|---|
| 0.02 | 0% | 1.00 | — | — |
| 0.05 | 0% | 1.00 | — | — |
| 0.10 | 10% | 1.15 | 0.30 | 0.00 |
| 0.15 | 48% | 1.85 | 0.30 | 0.00 |
| 0.20 | 71% | 2.69 | 0.63 | 0.00 |
| **0.30** | 90% | **4.67 ≈ GPT-4** | 0.81 | 0.00 |
| 0.50 | 98% | 9.04 | 0.96 | 0.63 |

**Random-floor:** OT beats random-at-matched-chunks at τ=0.20 AND τ=0.30 (vs JS which beat random at only τ=0.15).

**Per-sentence GPT-4-vs-OT (matched-chunk-count tau_ot per sentence):**
- **r(GPT-4, OT) = 0.306**, n=37 defined. Up from JS's 0.175 (n=48). Nearly doubled.
- Ours chunks_mean = 3.27 (vs GPT-4's 4.06). Delta mean_abs = 1.42.
- Ours Pearson_med = 0.794.

**Reordering candidates (top-8 lowest GPT-4 Pearson):** 3 MATCH, 5 MISS. But 4/5 MISS are single-chunk collapse (OT stays above τ=0.50 on those hard cases — coverage limit, not signal defect). Same idx=553850 catch as JS Config C, plus idx=493988 improves 0.81 → 0.66.

**Read.** H5 SUPPORTED. OT beats JS on two independent metrics (broader beats-random range; per-sentence r(GPT-4, ours) 0.175 → 0.306). Embedding-grounded cost earns its keep. Follow-up: extend τ grid to {0.70, 1.0} to close the 4 single-chunk-collapse cases; run topk / eps sensitivity ablation. All follow-up outputs at `results/phase1_tau_sweep_ot/{random_floor, per_sentence_compare}.json`.

### [DECISION] 2026-08-15 — Use `pot`'s `sinkhorn_log` for OT (was: hand-rolled log-Sinkhorn)
**Context:** User pointed to `https://pythonot.github.io/` after OT criterion was first implemented with a hand-rolled log-domain Sinkhorn. `pot 0.9.7.post1` was already installed via `create-venv.sh`.
**Options:** (a) keep hand-rolled; (b) switch to `pot.bregman.sinkhorn_log` (log-stabilised); (c) use `pot.sinkhorn2` with `method='sinkhorn_log'`.
**Chose:** (b). Cleaner code, standard citation, log-stabilised for small `eps`, returns transport plan (cost is `(T*C).sum()`). Torch tensors on GPU work natively. Cancelled queued job 176307109 (before it started) and resubmitted as 176307323.
**Verified:** OT values on the toy 3×3 test match hand-rolled to 4 decimal places on real data (0.6352 vs 0.6352; 0.0001 vs 0.0001). Correctness identical; library maintenance and citability better.
**Revisit if:** OT sensitivity ablations (topk, eps) reveal a bug or convergence issue that `pot`'s default settings don't handle.

### [RUN] 2026-08-15 — phase1_tau_sweep_base 176304944.gadi-pbs — completed
**Config:** backbone gemma-4-E2B **(base, not -it)**, same 48 sentences (seed 42, max_src_tokens=80). Criterion JS. Tau grid {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}. `--prompt_mode raw` (matches METHOD §1 spec — no chat template, base pretraining distribution). `--record_entropy`. Walltime 00:30:00 on 1×H200 gpuhopper.
**Command:** `qsub jobs/phase1_tau_sweep_base.pbs` (staged and fired immediately after `download_gemma4_e2b` completed).
**Result:** Ran on gadi-gpu-h200-0019; model load 28.3s, annotate 63.8s (~1.3s/sentence, 48/48 kept). Full sweep:

| tau | fire% | commit% | ours_ch | gpt4_ch | Pearson med | Pearson min |
|-----|-------|---------|---------|---------|-------------|-------------|
| 0.02 | 6% | 2% | 1.12 | 4.06 | 0.39 | 0.30 |
| 0.05 | 52% | 43% | 2.19 | 4.06 | 0.33 | 0.00 |
| **0.10** | 79% | 79% | **3.46** | 4.06 | **0.73** | 0.00 |
| 0.15 | 92% | 91% | 6.04 | 4.06 | 0.84 | 0.00 |
| 0.20 | 94% | 94% | 7.62 | 4.06 | 0.94 | 0.00 |
| 0.30 | 98% | 98% | 10.04 | 4.06 | 0.97 | 0.78 |

**Random-floor on base matrices:**

| tau | JS_med | RD_med | JS beats RD? |
|-----|--------|--------|--------------|
| 0.10 | 0.732 | 0.699 | no (barely loses) |
| **0.15** | **0.842** | **0.881** | **YES** (first observation ever of JS beating random) |
| 0.20 | 0.936 | 0.923 | no |

**Per-sentence GPT-4-vs-ours comparison (at per-sentence matched-chunk-count tau):**
- Ours chunks_mean = 2.96 (vs GPT-4 4.06). Chunk-count delta mean_abs = 1.44 (was 2.25 under -it+chat).
- Ours Pearson_med = 0.778 (vs -it+chat 0.919). Less diagonal.
- **Per-sentence r(GPT-4, ours) = 0.175** — barely improved from -it+chat's 0.149, but qualitative catch on reordering cases is real (see below).

**Catch on the top reordering candidate — idx=553850 (verb-final case):**
- GPT-4: 2 chunks, commit trace `[42×24, 53×6]`, Pearson=0.693. Reads almost the whole source before committing.
- **Ours (base + raw, matched-count tau): 2 chunks, Pearson=0.311.** Matches GPT-4's late-commit pattern. Compare -it+chat which gave 7 chunks with Pearson=0.907 (a MISS).

**Read:**
- Hypothesis (prompt confound is (part of) the story) is **partially supported**. Base+raw materially changes behaviour on reordering sentences; JS beats random-at-matched-latency at τ=0.15 (first time observed); chunk counts closer to GPT-4 than under -it+chat.
- Aggregate per-sentence r stays weak (0.175) because most sentences are monotonic and small commit-trace differences dominate the correlation. **The r-metric is not the right primary signal.** What matters is: on the sentences that GPT-4 identifies as non-monotonic, does ours also identify them as non-monotonic? Answer under base+raw: yes for idx=553850 (walked example). Need to walk the other reordering candidates to confirm.
- The tau=0.15 sweet spot: 92% fire, 6 chunks (moderately finer than GPT-4's 4), Pearson_med=0.84, and beats random. This is the first configuration that clears the "JS has signal" floor.
- **RWTH is still the arbiter** — Eq. 4 A-score is the primary metric we care about, and it cannot be computed on WMT training data. The manual RWTH fetch is now the top-priority external blocker.
- Do not yet claim "backbone-derived tags match GPT-4" — that needs RWTH. But we now have defensible tags to test against RWTH when the data lands.

### [DECISION] 2026-08-15 — Switch primary backbone from -it to base (gemma-4-E2B)
**Context:** Phase-1 tau sweeps under gemma-4-E2B-it exposed a prompt confound: raw-concat `{src}\n{tgt}` made JS *anti-signal* (worse than random-at-matched-latency) because the -it model treats raw concat as document continuation, not translation. Chat template fixed the fire-rate (22%→100%) but per-sentence r(GPT-4, ours)=0.15 — we catch different structure than GPT-4. Dipankar's suggestion: use the base pretrained model, where raw next-token prediction IS the natural task.
**Options:** (a) stick with -it + chat template; (b) switch to base + raw concat.
**Chose:** (b). Rationale: (1) matches METHOD §1's spec (P_pre / P_full are raw pretraining distributions, no task prompt); (2) removes the instruction-tuning confound at its root rather than papering over with prompt engineering; (3) same-model principle still holds — annotate with base, SFT with base; (4) cleaner story in the paper — no need to defend a chat-template choice.
**Trade-off:** Phase 2 SFT will start from a base checkpoint, so achieving translation quality will need more training epochs than starting from -it. Acceptable — Gate 1 (annotator quality) is upstream of Gate 3 (SFT quality) and doesn't depend on translation absolute quality.
**Verified before deciding:** `google/gemma-4-E2B` exists on HF, is the pretrained base (2.3B effective params, ~5GB safetensors), same architecture as the -it variant. Download job 176304709 fired on copyq.
**Revisit if:** base + raw concat's per-sentence r(GPT-4, ours) is no better than -it + chat (i.e., ~0.15), which would push us to blame the criterion (JS) rather than the prompt/backbone axis — trigger OT.

### [ANALYSIS] 2026-08-15 — Per-sentence GPT-4-vs-ours comparison on chat matrices
**Input:** `results/phase1_tau_sweep_chat/matrices.jsonl` (48 sentences under Gemma chat template).
**Scripts:** `scripts/phase1_gpt4_pearson.py` (GPT-4 baseline from shipped chunks); `scripts/phase1_per_sentence_compare.py` (per-sentence matched-tau comparison, r-of-Pearsons across sentences).

**Discriminating result — GPT-4 baseline:**
- GPT-4 Pearson_med **= 0.943** on same 48 sentences. min=0.693, max=0.984. Mean chunks/sentence = 4.06.
- WMT De→En at 30-50 tokens (after EAST App. C filter) is inherently monotonic. "Our criterion is diagonal" was NOT degeneracy — the ground-truth data is diagonal.

**Aggregate on ours (chat + JS at per-sentence matched-chunk-count tau):**
- Pearson_med **= 0.919**, min=0.313, max=0.982. Mean chunks = 5.98 (vs GPT-4's 4.06 — even strictest tau=0.01 produces finer chunks than GPT-4 on some sentences).
- Aggregate matches GPT-4 within noise.

**Per-sentence result — the key finding:**
- **Pearson-of-Pearsons across the 48 sentences: r = 0.149.** Our per-sentence Pearson does NOT track GPT-4's per-sentence Pearson.
- On the 8 lowest-GPT-4-Pearson sentences (reordering candidates): 5 MATCH (ours also < 0.85), 3 MISS.

**MISS case walked (idx=553850, high latency):**
- GPT-4: 2 chunks. Commit trace `[42×24, 53×6]` — reads 42 of 53 source tokens before committing anything, then translates 24 target tokens; reads remaining 11 tokens, translates 6. Very late, very safe.
- Reason: German subject `Ausnahmen für Emittenten ... bieten` splits subject and verb across positions 1-42; GPT-4 waits for the verb `bieten` before knowing the sentence structure.
- Ours (JS, tau=0.01): 7 chunks, first commit at i=9 (`Ausnahmen für Emittenten` → "Exemption for issuers" — Gemma is confident on cognates). Then i=14, 29, 39, 48, 52, 52.
- Two different policies: GPT-4 conservative-late, ours fast-early. **Without RWTH, neither is provably wrong.** The MISS case is exactly the German verb-final construction CLAUDE.md predicts should distinguish us — GPT-4 catches it here, ours doesn't.

**Read:**
- Aggregate Pearson matching GPT-4's is a weak positive. Per-sentence r=0.149 says we're catching *different* structure, not the same structure.
- **RWTH is now genuinely necessary** — the intrinsic Eq. 4 metric is the only arbiter that can decide whether our early commits are unfaithful (a_i > g_i violations) or whether GPT-4 is over-conservative. Without ground alignment, the extrinsic Pearson comparison is inconclusive.
- **OT is now the natural next criterion.** METHOD.md §3 hypothesis: uncertainty among semantically-nearby candidates is committable; uncertainty among semantically-distant candidates isn't. On idx=553850 the model is confident about "Exemption" but not the sentence structure — an embedding-aware ground cost should distinguish. Whether it delivers on Gemma-4-E2B is empirical.
- The entropy-vs-JS chunk-count matched comparison still not clean; skipping until OT is in place — the ordering question (does the oracle help?) is worth revisiting with three criteria in the CRITERIA registry, not two.

**What this does NOT resolve:**
- Sample is 48 sentences; per-sentence r=0.149 with n=48 has wide CI. Bump to ~200 before drawing firm conclusions.
- Backbone choice not tested — Qwen3.5-2B may produce different per-sentence structure.

### [RUN] 2026-08-15 — phase1_tau_sweep_chat 176272966.gadi-pbs — completed
**Config:** as prior entry.
**Command:** as prior entry.
**Result:** Ran on gadi-gpu-h200-0006; model load 51.8s, annotate 66.0s (~1.4s/sentence for 48 kept). Full sweep:

| tau | fire% | commit% | ours_ch | gpt4_ch | Pearson med | Pearson min |
|-----|-------|---------|---------|---------|-------------|-------------|
| 0.02 | 100% | 95% | 7.19 | 4.06 | 0.93 | 0.49 |
| 0.05 | 100% | 100% | 8.02 | 4.06 | 0.94 | 0.56 |
| 0.10 | 100% | 100% | 9.10 | 4.06 | 0.95 | 0.60 |
| 0.15 | 100% | 100% | 9.60 | 4.06 | 0.96 | 0.55 |
| 0.20 | 100% | 100% | 9.73 | 4.06 | 0.96 | 0.55 |
| 0.30 | 100% | 100% | 10.02 | 4.06 | 0.97 | 0.78 |

Random floor on chat matrices: JS still barely loses to random (2pp gap, was 15pp under raw). Entropy-only sweep at H_tau=2.0 (matched chunk count ≈ 4.4): Pearson_med=0.90 — comparable to JS but chunk counts don't match cleanly for a direct verdict on "oracle doing work."
**Read:** Chat template fixed the fire-rate (0% → 100%) but Pearson stayed high because the data itself is diagonal (see GPT-4 baseline entry above). All follow-ups landed in `results/phase1_tau_sweep_chat/{random_floor.json, entropy_sweep.json, gpt4_pearson.json, per_sentence_compare.json}`.

### [ANALYSIS] 2026-08-14 — Random-at-matched-latency floor on raw-concat matrices
**Input:** `results/phase1_tau_sweep/matrices.jsonl` (48 sentences, JS matrices under raw-concat prompt).
**Script:** `scripts/phase1_random_floor.py` — for each tau, samples 100 monotone random commit traces per sentence with the exact chunk-count JS produced at that tau, computes per-sentence mean Pearson(i*/n, j/m), then aggregates across sentences.
**Result:** JS Pearson_median > random Pearson_median at EVERY tau in the grid:

| tau | JS_med | JS_min | RD_med | RD_min | JS beats random? |
|-----|--------|--------|--------|--------|------------------|
| 0.02 | 0.33 | 0.25 | 0.00 | 0.00 | no |
| 0.05 | 0.53 | 0.28 | 0.00 | 0.00 | no |
| 0.10 | 0.82 | 0.22 | 0.69 | 0.00 | no |
| 0.15 | 0.86 | 0.00 | 0.79 | 0.00 | no |
| 0.20 | 0.92 | 0.00 | 0.89 | 0.00 | no |
| 0.30 | 0.96 | 0.42 | 0.93 | 0.00 | no |

**Read:** JS-derived commit points on Gemma-4-E2B (raw-concat prompt) are systematically **more diagonal** than uniform-random with matched chunk count — the criterion is *anti-signal* under this prompt. Consistent with the advisor's confound diagnosis: the model isn't doing translation on `{src}\n{tgt}`, so JS(P_pre, P_full) is tracking source-length accumulation, not translation committability. Do not conclude "JS is degenerate on Gemma-4-E2B" until the chat-template re-run lands. Note also that some sentences produce Pearson=0 (rows commit at nearly one position) — those are the outliers worth eyeballing regardless of aggregate.

### [RUN] 2026-08-14 — phase1_tau_sweep 176267898.gadi-pbs — completed
**Config:** backbone Gemma-4-E2B-it, data SiMT-De-En-660K (51 sentences balanced across latency, max_src_tokens=80, seed 42). Criterion JS (Jensen-Shannon, nats). Tau grid {0.02, 0.05, 0.10, 0.15, 0.20, 0.30} — evaluated offline from a single per-sentence full divergence matrix. Prompt mode raw-concat (`{src}\n{tgt}`). Walltime 00:30:00 on 1×H200 gpuhopper.
**Command:** as prior entry.
**Result:** Ran on gadi-gpu-h200-0017; model load 41.5s, annotate 62.1s (~1.3s/sentence, 48/51 kept). Full sweep:

| tau | fire% | commit% | ours_ch | gpt4_ch | Pearson med | Pearson min | #NaN |
|-----|-------|---------|---------|---------|-------------|-------------|------|
| 0.02 | 8% | 2% | 1.10 | 4.06 | 0.33 | 0.25 | 44 |
| 0.05 | 23% | 10% | 1.67 | 4.06 | 0.53 | 0.28 | 37 |
| 0.10 | 52% | 39% | 3.31 | 4.06 | 0.82 | 0.22 | 23 |
| 0.15 | 69% | 58% | 4.19 | 4.06 | 0.86 | 0.00 | 15 |
| 0.20 | 77% | 70% | 5.92 | 4.06 | 0.92 | 0.00 | 11 |
| 0.30 | 94% | 88% | 8.58 | 4.06 | 0.96 | 0.42 | 3 |

**Read:** Pearson_median rises monotonically with tau; getting fire coverage costs diagonal-bias. Chunk-count parity with GPT-4 (~4.1) lands at tau≈0.15 but Pearson_med there is 0.86. Combined with the random-floor analysis above (JS is beaten by uniform-random-at-matched-latency at every tau), the raw-concat prompt is confounded — the criterion is measuring "source-language token accumulation" more than "translation committability." Fix and re-run before drawing method-level conclusions. See the follow-up entry (phase1_tau_sweep_chat 176272966).

### [RUN] 2026-08-14 — phase1_smoke_js 176261302.gadi-pbs — completed
**Config:** backbone Gemma-4-E2B-it, data SiMT-De-En-660K (51 sentences balanced across latency, max_src_tokens=80, seed 42). Criterion JS (Jensen-Shannon, nats). Tau grid {0.02, 0.05, 0.10, 0.15, 0.20, 0.30} — evaluated offline from a single per-sentence full divergence matrix (annotator extended with `return_full_matrix=True`). Walltime 00:30:00 on 1×H200 gpuhopper.
**Command:** `python scripts/make_job.py --name phase1_tau_sweep --queue gpuhopper --ngpus 1 --walltime 00:30:00 --script "python scripts/phase1_tau_sweep.py --n_sentences 51 --criterion js --taus 0.02,0.05,0.10,0.15,0.20,0.30 --max_src_tokens 80" --output jobs/phase1_tau_sweep.pbs && qsub jobs/phase1_tau_sweep.pbs`
**Result:** QUEUED — awaiting run.
**Read:** Motivated by the previous smoke (tau=0.05 fired on only 22% of sentences). This sweep locates a tau range where the criterion actually fires across most sentences, and simultaneously flags positional-degeneracy at each tau by tracking Pearson(i*/n, j/m). The recorded matrices persist under `results/phase1_tau_sweep/matrices.jsonl` — future criterion swaps (KL, OT) and finer sweeps re-use the same forward passes.

### [RUN] 2026-08-14 — phase1_smoke_js 176261302.gadi-pbs — completed
**Config:** as above (21 requested → 18 kept after max_src_tokens=80 filter). JS, tau=0.05.
**Command:** as above.
**Result:** Ran on gadi-gpu-h200-0016; model load 30.6s, annotate 35.7s (~2.0s/sentence). **Fire fraction: 22% (4/18 sentences).** Of those four, Pearson(i*/n, j/m) values were 0.281, 0.955, 0.884, 0.534 — mean chunks_ours=1.72 vs chunks_gpt4=3.89. Fourteen of eighteen sentences collapsed to a single chunk because JS never dropped below 0.05.
**Read:** The mechanism works (structural checks all green; commit points where they fire are non-trivial). Threshold is the issue: JS ∈ [0, 0.693] and 0.05 is very strict for Gemma-4-E2B's predictive-distribution shifts on typical WMT De-En sentences. Sweep tau to find where fire fraction is well above 0 and Pearson isn't near 1 — that's the follow-on tau-sweep run (176267898). Do NOT scale to E4B yet — Gate 1 signal is not decidable from a threshold this tight.

### [DECISION] 2026-08-14 — RWTH gold alignments: URL confirmed, manual fetch step

### [DECISION] 2026-08-14 — Primary backbone switched: Gemma-4-E2B-it (was Qwen3.5-2B)
**Context:** Second session. User request: run the experimental programme on the Gemma-4 family, starting small and scaling. Both Gemma-4 sizes (`gemma-4-E2B-it`, `gemma-4-E4B-it`) are already on `MODEL_BASE` (see HOUSEKEEPING §5). This overrides the earlier same-day entry ("Primary backbone: `Qwen3.5-2B`") and HOUSEKEEPING §5 "Primary backbone" row.
**Options:** (a) keep Qwen3.5-2B as primary and Gemma-4 as ablation partner (unchanged); (b) swap — Gemma-4-E2B primary, Qwen3.5-2B ablation partner; (c) run both families as co-primaries.
**Chose:** (b). METHOD §5 same-model principle stays intact: annotate with Gemma-4-E2B → SFT Gemma-4-E2B. Ladder is E2B first, E4B only after Gate 1 passes on E2B (matches user's "start small, then scale"). Cross-family annotator-ablation partner becomes Qwen3.5-2B, matched at ~2B so the ablation still isolates family rather than scale. (c) rejected: doubles compute for a 14-week project and the primary claim only needs one backbone.
**Revisit if:** Gemma-4-E2B's `i*[j]` traces are degenerate under the METHOD §8 sanity checks (commit points cluster at sentence end, or `i*[j]/n ≈ j/m`). Fall back to Qwen3.5-2B and log the switch. Also revisit if Gemma-4's forward-pass path in the shared venv (`torch 2.11 + transformers 5.14`) turns out unstable — that would trigger a version-bump conversation with the `first-impressions-last` owner rather than a silent bump.
**Verified before deciding:** `AutoConfig.from_pretrained` + `AutoTokenizer.from_pretrained` both succeed on `gemma-4-E2B-it` under the shared venv (model_type=`gemma4`, text_vocab=262144, 35 text layers). End-to-end forward-pass load is the next smoke — see task list.

### [SESSION HANDOFF] 2026-08-14 — end-of-session state

**Repo:** clean, on `main` at `9e120cb`, synced with `github.com/dipankarsrirag/simt-tor-26`.

**Docs written this session:** `CLAUDE.md` (dataset roles table + WMT test-set section), `METHOD.md`, `EXPERIMENTS.md` (Stage-I scope, WMT22 correction from Ar/Zh error), `TIMELINE.md` (Phase 0 concrete deliverables + Stretches A/B/C), `RELATEDWORKS.md` (two-stage recipe), `HOUSEKEEPING.md` (paths, compute, git, data table, venv discipline), `LOG.md` (this file), `OPTIONALS.md` (venue verdict, 3 blockers, 4 strengthening, 7 method improvements, closest-work distinctions, 2×2 novelty frame).

**Infrastructure scaffolded:** `.gitignore`, `create-venv.sh` (not yet run), `scripts/make_job.py` (gpuhopper+copyq only, shared `/g/data/po67/dipankar/cache/`), `pbs/env.sh`, `pbs/templates/job.pbs.tpl` (auto-resubmit), `src/constants.py`, `src/{annotator,train,eval}/`, `scripts/download_data.sh`, `data/` symlink to `/g/data/po67/dipankar/data/simt-tor-26/`.

**Pending — needs human decision before Phase 0 code starts:**

1. **Scale framing.** OPTIONALS.md §Blocker 1: Option A ("at 2B" preregistered) vs Option B (post-writeup 8B replication on `Llama-3.1-8B-Instruct`). Recommendation A. Blocks the paper's abstract wording; not blocking Phase 0 code.
2. **OPTIONALS.md method-improvement scope.** Which of M1–M7 go in the annotator. Recommendation: M1, M2, M3, M5, M7 (High-priority set + trivial M5). Blocks the annotator design — decide before Phase 1.
3. **Paper name.** Suggested `DRIFT` (Distributional Read/write Inference-Free Training). Not blocking code, but easier to fix before project-name strings enter scripts.

**Pending — infrastructure work not blocked on human decision:**

4. **RWTH De→En gold alignments URL.** `scripts/download_data.sh` step 5 is a TODO placeholder. EAST paper §E.4 has the source. Once URL is in, re-run `qsub jobs/download_data.pbs` (idempotent — will only fetch RWTH). Blocks the Gate 1 intrinsic annotation-quality measure.
5. **`bash create-venv.sh` — layers `pot / trl / accelerate / peft / datasets / sacrebleu` onto the shared `.venv-fil`.** Not yet run. Coordinate with `first-impressions-last` and `simul-mt` owners per HOUSEKEEPING §4.1 shared-venv discipline. Blocks any code that imports these packages.
6. **BLEURT-20 fetch to `MODEL_BASE/BLEURT-20/`.** Flagged in HOUSEKEEPING §5. Needed for the third-metric row in `EXPERIMENTS.md`. Trivial `copyq` job; not blocking early phases.
7. **`scripts/build_off_multi.py` — Off-Multi-120K assembly from WMT17-21 test data à la ALMA.** Only needed for Stretch A (multilingual Stage II), not for the primary Stage-I result.

**Context prime for next session.** Read order: `CLAUDE.md` (project spec + dataset table) → `OPTIONALS.md` (paper strategy; the 2×2 diagonal-move framing is the anchor) → `TIMELINE.md` Phase 0. Do not start writing the training pipeline — the annotator is the project, the SFT is plumbing.

---

### [RUN] 2026-08-14 — copyq download job 176225855.gadi-pbs
**Config:** copyq, 1 CPU / 8 GB / 100 GB jobfs, walltime 04:00:00. Job script `jobs/download_data.pbs` calls `scripts/download_data.sh`.
**Command:** `qsub jobs/download_data.pbs`
**Result:** `SiMT-De-En-660K` (660,876 rows, 685 MB — latency counts: low=230,902 / medium=227,131 / high=202,843), `SiMT-Multi-90K` (67 MB, 8 directions), WMT15 De-En newstest2015 (2,169 sentence pairs, 504 KB), WMT22 all 8 pairs `{de,en,zh,ru,cs}-{en,de,zh,ru,cs}` with `docid` (3.9 MB). RWTH and Off-Multi-120K skipped (TODOs). Log at `logs/download_data.log`.
**Read:** All Stage-I data assets are on disk at `/g/data/po67/dipankar/data/simt-tor-26/`. `data/` symlink from the repo resolves. Ready for Phase 0 format inspection and Phase 1 annotator development. RWTH still needed for Gate 1 intrinsic eval.

---

### [DECISION] 2026-08-14 — Scope: Stage I only; Stage II is stretch
**Context:** EAST is a two-stage recipe (§3.2 of the paper): full-weight SFT on `SiMT-De-En-660K` (Stage I, De→En) then LoRA on `SiMT-Multi-90K` + `Off-Multi-120K` (Stage II, 8 directions). Our 14-week timeline with a 2B backbone cannot cover both properly.
**Options:** (a) Stage I only, matched comparison at De→En. (b) Stage I + Stage II subset, sacrificing ablation depth. (c) Full recipe on a smaller data subset each — matches EAST shape but neither stage lands cleanly.
**Chose:** (a). The claim lives in the annotation criterion, which decides tag placement in Stage I; Stage II just LoRA-adds on top of Stage-I tags and can't move the criterion. EAST publishes Stage-I numbers separately (Figure 3 "EAST-Stage-I"), giving us a matched target. Stretches A, B, C in `TIMELINE.md` are the multilingual, document-level, and conversational extensions — all gated on Gate 3.
**Revisit if:** the Stage-I result lands early (say by week 8) with room to spare, and Dipankar wants to add multilingual before the writeup.

### [DECISION] 2026-08-14 — Primary backbone: `Qwen3.5-2B`
**Context:** EAST's Table 2 uses Llama-3-8B-Instruct. Our compute is one H200 per job (see `HOUSEKEEPING.md` §6), which comfortably fits 2B full-weight tuning with margin for the annotator's prefix-batch passes. Larger backbones would eat Phase 2 walltime that we need for `tau` sweeps and ablations.
**Options:** (a) `Qwen3.5-2B`, (b) `gemma-4-E2B-it`, (c) 4B variants of either.
**Chose:** (a) as primary, (b) as the cross-family annotator-ablation partner. Sizes matched at 2B so the annotator-model ablation isolates family, not scale. Scale-up to 4B stays available (both on disk) if Gate 3 passes with headroom.
**Revisit if:** `METHOD.md` §8 sanity checks show `Qwen3.5-2B` produces degenerate `i*[j]` traces (commit points cluster at sentence end). Then switch to `gemma-4-E2B-it` and re-check.

### [DECISION] YYYY-MM-DD — Annotator is the same model as the fine-tuning backbone
**Context:** EAST uses GPT-4 as an external annotator. We need to decide whether to self-annotate or use a larger teacher.
**Options:** (a) same model, (b) larger external annotator, (c) GPT-4 as in EAST.
**Chose:** (a). Cleaner claim — no external teacher, no distillation dependency, and tags are calibrated to the model that must act on them. A larger annotator would likely give better tags but reintroduces exactly the dependency we are criticising.
**Revisit if:** the cross-annotation ablation shows same-model annotation underperforms — that would mean error amplification dominates self-calibration.