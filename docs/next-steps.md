# Next steps, in order — Week-by-week critical path to Findings-tier submission

Written 2026-08-18. **Updated 2026-08-22** — the plan below (Weeks 1-8) is largely obsolete; the Aug 21 v6 pivot + Aug 22 v6b work have superseded most of it. Read the "As of 2026-08-22" block first; refer to Weeks 3-8 only for late-stage / writing tasks that are still valid.

Target: ACL/EMNLP Findings; IWSLT hedge.

## As of 2026-08-22 — current state + next actions

**Ship model:** `_archive/results/gemma_2b_curated/sft_multilingual_v6b_ctrl_merged3/final/` (Gemma-4-E2B-it + chat template + NL latency + direct-ids splice + α=1 + EAST §3.1 merge at <=3-word threshold). Trained on 79K rows × 8 language pairs. See `00-README.md` "Naming" table for the full family of trained arms.

**Headline sanity numbers (N=50 FLORES devtest, mean across 40 cells = 8 dirs × 5 latencies):**

| variant | mean BLEU | mean AL | mean chunks/sent |
|---|---|---|---|
| ctrl (raw OT) | 24.89 | 3.32 | 10.5 |
| merged (<2 words) | 27.70 | 3.46 | 7.2 |
| **merged3 (<=3 words) — ship** | **29.46** | **4.78** | **4.5** |
| E4B on raw OT | 28.10 | 3.92 | 8.2 |
| cond-A (GPT-4 chunks, 4 dirs only) | 30.51 (20 cells) | 5.69 | 3.8 |

On de-en at low_medium latency, **merged3 (31.88) beats cond-A (30.90)** — matched-backbone head-to-head win. E4B on raw OT underperforms merged3 → chunk simplification > model scaling for this task.

**Immediate next actions, in order:**

1. **Full N=1012 FLORES on merged3** (biggest gap in evidence right now). Sanity is N=50; publication-scale requires the full devtest. Re-use `jobs/phase2_extrinsic_stream_v6b_merged_sanity_TEMPLATE.sh merged3` pattern with `N=1012` and 4h walltime. 5 latency PBS jobs × 8 dirs = 40 cells.
2. **Full N=1012 WMT15 De↔En on merged3** for EAST Fig 3 head-to-head. WMT15 test set already downloaded at `/g/data/ba39/dipankar/simul-mt/data/eval/de-en/wmt15.*`. 1 PBS with 5 latencies × 2 dirs, ~4-5h.
3. **Combined E4B + merged3 SFT** — stacking test. Same v6b-ctrl-merged3 recipe on Gemma-4-E4B-it (2× params). Expected: closes remaining ~1 BLEU gap to cond-A, or exceeds it. Data is `sft_dataset_multilingual_v6b_merged3.json` (79K rows), model_path swap only. Uses `sft_v6.py` with `--per_device_batch_size 8 --grad_accum_steps 8` (same effective batch, halved per-device for E4B memory). ~2h GPU.
4. **Optional: Fair scaling test** (E4B annotator → E4B chunks → E4B SFT). Needs the sentence-batched annotator (15× speedup verified but ~3-4% divergence from naive; see LOG `[DECISION] 2026-08-22 — Annotator KV cache & sentence batching reject byte-identical`). Cost with batching: ~2-4h GPU for annotation + 2h SFT + 30min sanity. Skip if the "combined E4B+merged3" is enough for the paper.

**Legacy plan (Weeks 1-8) below is retained for context but has been superseded.** The v6 pivot (2026-08-21) + v6b fixes (2026-08-22) restructured the work: we now train on multilingual v6b (8 dirs) rather than the per-backbone {E2B, E4B, Qwen35} single-dir n=10K runs the old plan called for. Gates A/B are moot — the paper's claim is now the merged3-vs-cond-A matched head-to-head at 4 language pairs + coverage extension to ar/vi where cond-A can't reach.

## Where we are (Week 0 = current, legacy plan — SUPERSEDED)

- **Phase 1 (annotator)**: DONE. OT + τ=0.30 on base Gemma-4-E2B + raw concat. Gate 1 passed n=210 stratified.
- **Phase 2 headline (E2B, n=10K)** (historical / cond-A archaeology; cond-A removed 2026-08-18): OT-SFT (formerly cond-B) beat cond-A by +4.8-5.7 BLEU across wait_k∈{3,5,7} at matched AL, on newstest2013. That specific vs-GPT-4 delta is now historical — new comparison is OT-SFT vs published-competitor numbers (Fig. 1/2).
- **Completed this session (2026-08-18 late)**:
  - Qwen3.5-2B OT annotation COMPLETE (9,550/9,550 sents); dataset built as `sft_dataset_n10k_annotator-qwen35.json`.
  - E4B OT annotation COMPLETE (9,567/9,567 sents, 20 shards).
  - Extended wait-k (k∈{1,9,11}) cond-A results landed; cond-B extended wait-k was walltime-killed at ~2675/3000 (queued for re-submit as split jobs).
  - Latency-prompt sweep (low/high) landed for both arms — null result, BLEU swings ≤0.5 across latency prompts at n=10K.

## The two gates that determine venue

| Gate | Hypothesis | What passes | Impact if passed | Impact if failed |
|---|---|---|---|---|
| **A** | P2 (Qwen cross-family) | Qwen OT-SFT beats past-work published wait-k=5 De→En number by ≥ +2 BLEU | Findings 65-80% | Findings 25-35%; retreat to IWSLT |
| **B** | P1 (vs Simul-LLM published) | OT-SFT beats Simul-LLM's published wait-k=5 De→En BLEU by ≥ +2 | Findings 70-85% | Findings 15-25%; rebuttal-cycle within-framework WaitK-SFT is the escape hatch |

Gate B is the **highest-risk gate** — if OT-SFT ties Simul-LLM's published number, the "chunk quality matters" story dies without a within-framework ablation to fall back on. If a reviewer objects that "your +BLEU vs Simul-LLM comes from EAST framework overhead, not from OT chunks," we build the within-framework WaitK-SFT arm in the ~2-day rebuttal window (Cond-C was removed 2026-08-18 late per user pivot to compare-against-past-work-verbatim). See LOG for the removal decision.

## Acceptance-probability table

Calibrated from analogous 2024-25 SiMT/LLM-MT papers at each venue:

| Venue | Baseline (H8 only) | + Gate A pass | + Gate B pass | Gate A fail | Gate B fail |
|---|---|---|---|---|---|
| ACL/NAACL **Main** | 10-20% | 15-25% | 20-30% | <5% | <5% |
| ACL/NAACL **Findings** | 55-70% | 65-80% | **70-85%** | 25-35% | 15-25% |
| **COLING** main | 70-85% | 80-90% | 85-92% | 45-55% | 30-40% |
| **IWSLT** system | 90-95% | 93-96% | 95-97% | 80-90% | 70-85% |

**Realistic best case (if both gates pass + reordering-subset supports H11 + multi-seed lands):** ARR January cycle Findings 70-85%. Main-track push (10-20%) is a lottery ticket, not a plan.

---

## Week 1 — Retrain OT-SFT on v2 dataset + Test A adaptivity probe

Two parallel workstreams. Cond-C and Cond-D are gone (see LOG `[DECISION] 2026-08-18 late — Remove Cond-C entirely`); we now compare against past-work published numbers verbatim. Gate B is redefined as OT-SFT vs Simul-LLM's published wait-k=5 De→En BLEU.

**Status (2026-08-19):** 1a SFT ✓ landed (`sft_n10k_v2/final`, eval_loss 1.632); 1a streaming eval Q (176597836); 1b code ✓ shipped in `src/eval/extrinsic.py`, smoke ✓ passed + preliminary null on adaptivity, 3-family sweep Q (176599155/6/7); 1c ✓ DONE (WMT15 34.24, WMT22 28.60).

### 1a. Retrain OT-SFT/n=10K on `sft_dataset_n10k_v2.json` — SFT DONE 2026-08-19, streaming eval queued

Dataset built with 2026-08-18 fixes: fallback τ ladder + latency reassignment. Stats: 9,562 kept / 5 skipped, collapse rate down from 28% (v1) to 0.05% (v2), 71.7% used primary τ=0.30, latency distribution 57% low / 20% medium / 23% high.

**SFT LANDED (job 176597831, 2026-08-19).** 36.9 min wall, early-stopped at step 700 / epoch 1.23, best `eval_loss = 1.632` (v1 was 1.677). All 5 EAST-token embeddings moved ~0.08 L2. Sample gens emit clean `<|end-of-read|>` / `<|end-of-write|>`. `_archive/results/gemma_2b_curated/sft_n10k_v2/final/` is the checkpoint.

**Streaming eval queued (176597836, chained afterok on SFT).** Runs the 4 policies (wait_k ∈ {3,5,7} + check_argmax) at 3000 sents on newstest2013. Predicted: chunks/sent > 1 under check_argmax if collapse rows were the bottleneck (P3 sub-claim iv). AL under check_argmax drops from v1's ~18 to a plausible mid-latency range. Now uses lean 4h30m walltime + per-policy skip-if-exists + chain-at-start (see LOG `[DECISION] 2026-08-19 — Week-1 PBS walltimes shortened`).

### 1b. Test A — soft commit criteria at inference (adaptivity probe) — CODE SHIPPED 2026-08-19; sweeps Q

Added three policies to `src/eval/extrinsic.py::stream_translate` (SHIPPED):
- `check_prob_thresh` — commit if `p(EOR | context) > θ` for θ ∈ {0.05, 0.10, 0.20}.
- `check_rank` — commit if `rank(EOR) ≤ r` (min-rank-on-ties) for r ∈ {1, 2, 3, 5}.
- `check_ratio` — commit if `p(EOR) / p(top_non_eor) > k` for k ∈ {0.1, 0.5, 1.0}.

Sanity checks (advisor 2026-08-18): `check_ratio 1.0 == check_argmax` mathematically; `check_rank 1 == check_argmax` (min-rank convention). Eyeball once results land.

**Smoke PASSED (176597830, 2026-08-18 late, 20 sents on `sft_n10k/final`):** BLEU 41.93 (non-zero ✓), AL 17.85 finite ✓. **BUT** `p(EOR) > 0.10` never fired across all 20 sents — every g_words vector uniformly equals src_words (single chunk drain at source-exhaust, matching v1's `check_argmax` degeneracy). This is the preliminary null on "hard argmax is hiding adaptivity" — Test A's whole claim to fame.

**Full 10-config sweep Q on `sft_n10k/final`** (v1 checkpoint, no retrain): 176599155 (thresh 3 configs, 3h walltime), 176599156 (rank 4 configs, 4h walltime), 176599157 (ratio 3 configs, 3h walltime). All lean walltime + per-config skip + chain-at-start (MAX_SHARDS=2 each).

### 1c. WMT15 + WMT22 De→En reruns — DONE 2026-08-19

Offline BLEU on `sft_n10k/final` (v1 checkpoint):

| Test set | n | BLEU | s/sent | Feeds |
|---|---|---|---|---|
| WMT15 newstest2015 | 2169 | **34.24** | 0.56 | Fig. 1 axis (non-LLM competitors, AL) |
| WMT22 newstest2022 | 1984 | **28.60** | 0.48 | Fig. 2 axis (LLM competitors, LAAL); EAST Table 2 head-to-head |

Signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. WMT22 vs EAST Table 2 De→En 32.55: ~4 BLEU behind at 4× params × 66× data disadvantage. **Streaming BLEU on these same test sets remains unmeasured** — the paper story needs a follow-up streaming eval on WMT22 De→En for the full Fig. 2 axis. Queue after Week 1 lands.

---

## Week 2 — Qwen + E4B OT-SFT runs (Gate A + backbone-scale)

**Concrete work:**
1. **Qwen OT-SFT** on `sft_dataset_n10k_annotator-qwen35.json` (dataset built 2026-08-18). Streaming eval → Gate A result.
2. **E4B OT-SFT** — build dataset from `annot_ot_e4b_n10k/matrices.jsonl` using `scripts/08_build_sft_dataset.py` (2026-08-18 fixes apply); SFT; streaming eval → P2 (ii) result.
3. Optional Test B (loss upweighting on EAST specials) if Test A doesn't find adaptivity: add `--special_token_loss_weight` to `sft.py`; sweep α ∈ {3, 5, 10}; retrain OT-SFT.

**Deliverable:** Gate A pass/fail; P2 (ii) result on 4B params.

---

## Week 3 — Reordering-subset analysis + cross-annotator SFT starts

**Why now:** Reordering-subset is a P1 sub-claim (mechanism); no GPU compute needed. Cross-annotator (P4) needs all three OT annotations done — that already lands end of Week 2.

**Concrete work:**
1. `scripts/phase2_reordering_stratified_bleu.py` — bin newstest2013 by per-sentence GPT-4-style Pearson using our own annotator (approximation OK). Bins: monotone (≥0.90), mild (0.70-0.90), reordering (<0.70). Report BLEU-vs-AL per bin from existing streaming eval JSONs. **Predicted:** OT-SFT lead over published wait-k baselines widens on reordering bin.
2. Start cross-annotator SFT matrix (6 off-diagonal cells):

| SFT backbone \ Annotator | E2B chunks | E4B chunks | Qwen chunks |
|---|---|---|---|
| E2B | ✓ done | queued | queued |
| E4B | queued | ✓ (self) | queued |
| Qwen | queued | queued | ✓ (self) |

Each off-diagonal: ~1h SFT + ~5h streaming eval on gpuhopper. Total ~36 hours GPU across 6 runs.

**Deliverable:** Fig. 3-bis (stratified bar chart per H11); one paper table row per matrix cell (H10).

---

## Week 4 — Data-scale curve on champion + τ-generalisation smoke [H14 + H18]

**Why now:** Cannot start scale until champion is identified (winner across E2B/E4B/Qwen from Week 2 results). τ-generalisation smoke can run in parallel.

**Concrete work:**
1. Identify champion: highest absolute BLEU at wait_k=5 among {E2B, E4B, Qwen} OT-SFT.
2. For champion backbone, annotate n=20K/30K/40K/50K (batched OT ~2-4s/sent × 50K = 28-56h split across shards).
3. Build OT-SFT dataset at each scale with 2026-08-18 fixes, SFT, streaming eval at wait_k=5.
4. **τ-generalisation smoke (H18):** for each Multi-90K direction {en-de, en-zh, en-cs, en-ru — both directions where applicable, 500 sents/pair × 4 τ values ∈ {0.10, 0.20, 0.30, 0.50}. Small toy-SFT + streaming eval. Confirms τ=0.30 works within ±10% of best per-pair τ.

**Deliverable:** Paper Fig. 4 = data-efficiency curve (H14) + Table on τ generalisation (H18) confirming τ=0.30 as fire-and-forget across languages.

---

## Week 5-6 — Multi-lingual mixed-corpus training [H19] (paper's strongest framing)

**Why now:** Direct head-to-head with EAST Table 2 (offline X↔En multi-lingual). Requires τ=0.30 to be validated by P2 (iv) first.

**Concrete work:**
1. Annotate 10K/pair × 4 pairs {en-de, en-zh, en-cs, en-ru} at fixed τ=0.30 on champion backbone. ~50 GPU-hours, splittable across shards.
2. Build MIXED OT-SFT dataset (40K rows total, latency-balanced across pairs) via `phase2_build_sft_dataset.py` (2026-08-18 fixes apply per-language).
3. SFT single OT-SFT model on mixed 40K corpus (~2-3h on champion backbone).
4. Streaming eval each pair on WMT22 X↔En test set, wait_k∈{3,5,7} + check_argmax. (~5h × 4 pairs = ~20h)
5. Also train per-pair OT-SFT models on 10K single-lang for the "mixed beats per-pair" comparison. ~1h × 4.

**Deliverable:** Paper's central multi-lingual table (mirrors EAST Table 2). Reports: (a) mixed OT-SFT vs EAST published numbers per pair; (b) mixed OT-SFT vs per-pair single-lang OT-SFT (does mixed help or hurt?); (c) τ=0.30 within 1 BLEU of per-pair optimal τ.

**Why Multi-90K's 4 pairs (not en-es/en-vi/en-ar):** Multi-90K test sets are available for all 4 pairs on the WMT22 X↔En benchmark that EAST also reports on — apples-to-apples with their Table 2. en-es/en-vi/en-ar were dropped 2026-08-18 (see LOG `[DECISION] 2026-08-18 — Multi-lingual via SiMT-Multi-90K`).

---

## Week 7 — AL-CA measurement + WMT test-set numbers (report ONCE)

**Why now:** AL-CA (`torch.cuda.Event` per emit-step) is the last engineering piece. Then run everything ONCE on the frozen test sets — those are the reportable numbers, no more tuning after.

**Concrete work:**
1. `src/eval/extrinsic.py::stream_translate` — add `torch.cuda.Event(enable_timing=True)` accumulator; discard first 30 sentences as CUDA warmup; report AL-CA per EAST Table 3 formula.
2. Validate Layer 3 AL-CA against `scripts/phase2_compute_al_ca_approx.py` corpus-level approximation on 500 sentences.
3. Rerun champion OT-SFT + mixed OT-SFT models on WMT22 De→En (for direct EAST Table 3 comparison) AND WMT22 X↔En for the multi-lingual table. Compute BLEU, COMET-22, AL, AL-CA. **Do not tune anything after this.**

**Deliverable:** Frozen test-set numbers. Paper table populated with head-to-head against EAST Tables 1, 2, 3.

---

## Week 8 — Writing sprint

**Concrete work:**
1. Draft in this order: (a) intro + 2×2 framing pitch (see `_archive/OPTIONALS.md`), (b) related work (Simul-LLM/TransLLaMa/AlignAtt/DaP-SiMT distinctions), (c) method (annotator + OT criterion + dataset construction + τ-generalisation claim per H18), (d) experiments (Table 1 matched De→En matrix, Table 2 multi-lingual mixed per H19 mirroring EAST Table 2, Fig. 3 BLEU-vs-AL, Fig. 4 data scale, Fig. 3-bis reordering-stratified), (e) discussion (H9 reframing + limitations).
2. Rebuttal-fodder: multi-seed on champion pair (if reviewers ask); RWTH-A intrinsic (Phase 3 appendix); en-ar reproduction as appendix if time.

**Target submission:** ARR March (multi-lingual expansion pushes January cycle). ACL 2027 commitment window follows March cycle.

---

## Decisions applied 2026-08-18 (from user pivots)

- **REMOVED: cond-A entirely** (2026-08-18 late). See `../LOG.md` `[DECISION] 2026-08-18 late — Remove cond-A entirely`.
- **REMOVED: Cond-C (within-framework wait-k baseline)** (2026-08-18 late). Redundant given past-work-verbatim comparison strategy; rebuttal-cycle rebuild path preserved. See LOG `[DECISION] 2026-08-18 late — Remove Cond-C entirely`. Live arm now: OT-SFT only.
- **DROPPED: multi-seed protocol.** Signal is large vs per-seed noise ~0.5 BLEU on WMT De→En at 10K rows. Add in rebuttal cycle if raised.
- **DROPPED: en-es / en-vi / en-ar as originally scoped.** Replaced by Multi-90K's 4 pairs (en-de/en-zh/en-cs/en-ru).
- **ADDED: H18 (τ generalisation across languages) and H19 (mixed-lingual training with single τ dominates per-pair).** See `02-hypotheses.md`.
- **Submission target:** ARR March (was January). Multi-lingual expansion is worth the extra 2 months given it maps directly onto EAST Table 2 for head-to-head.

## Immediate deliverables (this week, no new GPU compute needed)

- **EAST head-to-head numbers for the current De→En result.** OT-SFT offline BLEU 32.54 vs EAST 32.55 (Table 2 De→En) — statistical tie at 4×/66× disadvantage. Add COMET-22 to `sft_n10k/final/` (~30 min inference on H200).
- **Rerun offline BLEU on WMT22 De→En test set** (currently on newstest2013) for truly matched comparison to EAST Table 2. ~30 min per arm.
- Update paper draft's abstract with "matches EAST offline BLEU at 4× fewer params, 66× less data, Stage I only."

## Blockers, right now

- **Cond-C removed 2026-08-18** — no longer a Week-1 blocker. Rebuttal-cycle rebuild only if reviewers demand framework-controlled ablation.
- **Cond-D dataset builder not yet written** — Week 2 hard blocker.
- **Qwen annotation COMPLETE + dataset built.** Qwen OT-SFT ready to submit.
- **E4B annotation ~34% done** (2430/7095) — controls H12 timing. Self-resubmitting through MAX_SHARDS=40 gate.
- **AL-CA measurement not implemented** — corpus-level approximation exists; Layer 3 needs ~30 lines.
- **WMT22 De→En test set fetch verification** — data was downloaded but needs a quick smoke: `wc -l data/wmt22/*.de` and confirm one-to-one alignment with `.en`.
- **RWTH baseline decision** (GPT-4 API re-annotation) — Phase 3 appendix only, not blocking Findings.

## Not blockers (deferrable to post-writeup / rebuttal)

- Off-Multi-120K + Stage-II LoRA.
- 8B replication (`_archive/docs/OPTIONALS.md` §Blocker 1 Option B — post-writeup only).
- en-es / en-vi / en-ar — post-writeup stretch, appendix at best.
- Multi-seed + paired bootstrap — rebuttal cycle only.
- BLEURT-20 metric alongside sacrebleu — nice-to-have, not required for Findings.
- Doc-level and conversational SiMT.

## Weekly checkpoint reminder

Bring `../LOG.md` to Dipankar meetings, not a summary — `setup.md` §1.
