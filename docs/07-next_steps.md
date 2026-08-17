# Next steps, in order

Ordered by priority. Each item states what it does, why now, and what unlocks after. Written 2026-08-18 after Phase 2 headline result landed (H8 confirmed, H9 refuted).

## Where we are

- **Phase 1 (annotator)**: DONE. OT + τ=0.30 on base Gemma-4-E2B + raw concat. Gate 1 passed n=210 stratified.
- **Phase 2 (SFT + streaming)**: HEADLINE RESULT LANDED on Gemma-4-E2B, n=10K. Cond-B (OT) beats cond-A (GPT-4) by +4.8-5.7 BLEU across wait_k ∈ {3, 5, 7} at matched AL, on newstest2013 (3,000 sents).
- **In flight**: Qwen3.5-2B replication (H6), Gemma-4-E4B base replication (H7), extended wait-k / per-latency-prompt runs, matplotlib install for figure generation.

## 1. Complete the replication matrix (in flight — no new work required, monitor)

Once these land, we have the full "same headline result on 3 backbones" story that reviewers ask for.

| Backbone | cond-A SFT | cond-B annotation | cond-B SFT | Streaming eval |
|---|---|---|---|---|
| Gemma-4-E2B (2B) | ✓ done | ✓ done | ✓ done | ✓ done (paper Fig. 3) |
| Qwen3.5-2B (2B, cross-family) | ✓ done | 🔄 running (~54% at last check) | queued after annotation | queued after SFT |
| Gemma-4-E4B (4B, scale-up) | 🔄 running | 🔄 running | queued after annotation | queued after SFT |

**When each finishes, run the same 4-policy streaming smoke** (`scripts/phase2_extrinsic_streaming_smoke_condB.pbs` template, adjust MODEL_DIR).

**Unlocks:** Paper Table 1 = matched matrix (cond A/B × 3 backbones) at wait_k=5. Multi-backbone replication is expected by ACL Findings tier reviewers.

## 2. Extended BLEU-vs-AL curve on E2B (in flight — jobs 176531163/164)

Currently 3 wait-k points (k=3, 5, 7) + check_argmax. Extending to k ∈ {1, 9, 11} gives 6 wait-k points for a smooth trade-off curve on the paper's Figure 3.

**Unlocks:** matplotlib install (job 176531167) → `python scripts/phase2_plot_bleu_al.py` → PDF for paper.

## 3. Per-latency-prompt sweep (in flight — jobs 176531165/166)

Model was trained with `<|low-latency|>`, `<|medium-latency|>`, `<|high-latency|>`. Evaluating under each latency prompt reproduces EAST's Table 3 format (BLEU/AL per latency setting).

**Unlocks:** Second axis for the paper — the latency-prompt is one lever, wait-k is another, and their interaction is worth reporting.

## 4. Reordering-subset analysis (~1 day of scripting, no new GPU compute)

Already have `results/gate1/gpt4_pearson_full.json` (per-sentence Pearson on 660K sentences). Newstest2013 sentences are not in that corpus, but the same per-sentence Pearson can be computed on the 3,000 newstest2013 lines using the same script logic.

- Compute GPT-4-style per-sentence Pearson on newstest2013 (using our own tokenizer for chunking — approximation).
- Bin: monotone (≥0.90), mild (0.70-0.90), reordering (<0.70).
- Report BLEU-vs-AL per bin for cond-A and cond-B.

**Predicted (mechanism claim, H5-descendant):** cond-B's lead widens on the reordering bin. If confirmed, it's the paper's mechanistic sub-figure and directly connects Phase 1 (annotator picks up reordering) to Phase 2 (SFT capitalises on it).

## 5. AL-CA (Computation-Aware Average Lagging) — Layer 3 measurement

EAST Table 3 reports AL, AL-CA, and WWT (wall-time per word). Small addition to `src/eval/extrinsic.py::stream_translate`:

```python
import torch
event_start = torch.cuda.Event(enable_timing=True)
event_end = torch.cuda.Event(enable_timing=True)
event_start.record()
# ... generation step ...
event_end.record()
torch.cuda.synchronize()
elapsed_ms = event_start.elapsed_time(event_end)
```

Accumulate per-token wall-time. Discard first N sentences as CUDA warmup. Then:

```
AL-CA = (1/tau) * sum (g(i) + wall_time_ratio(i) - (i-1)*|X|/|Y|)
```

**Unlocks:** apples-to-apples with EAST's Table 3 latency numbers.

## 6. Scale training data on champion (10K → 50K)

Once winner across {E2B, E4B, Qwen3.5-2B} identified from the replication matrix (§1), run the SAME matched cond-A vs cond-B pipeline at:

- n=20K (~19K after filters)
- n=30K
- n=40K
- n=50K

Compute per point:
- Annotate cond-B (batched OT ~2s/sent on E2B, ~4s on E4B; 50K ≈ 28h wall on E4B, split across shards).
- Train cond-A + cond-B (~40min at n=10K on E2B; ~2h on E4B at n=50K).
- Streaming eval on newstest2013 (existing 5h job template).

**Unlocks:** paper's Figure 4 = data-efficiency curve. EAST reports theirs on Fig. 6 at 660K. Ours would show competitive performance at ~5% of their data.

## 7. Newstest2015 test-set numbers (report ONCE)

After all dev-set analysis is done and champion is picked, run the SAME extrinsic on newstest2015 (2,169 sentences). These are the paper's primary numbers. **No hyperparameter tuning after this run.**

## 8. Multiple seeds + paired bootstrap (statistical significance)

For the champion model at n=10K + n=50K, run each SFT with 3 seeds (42, 142, 242). Report BLEU as mean ± std across seeds. Paired bootstrap on the sentence-level BLEU differences to give a p-value on the A-vs-B claim. This is expected for ACL/EMNLP Findings.

## 9. RWTH intrinsic — Phase 3 appendix eval (EAST App. E.4 mirror)

Explained in `06-data.md`. Metric: `A = (1/T) Σ I[a_i ≤ g_i]` where `a_i` is the human-aligned source position for target word `i` and `g_i` is our commit position. Higher = more faithful; wait-until-end gets A=1 but at max latency.

**Baseline decision blocks this**: need GPT-4-API re-annotation of RWTH's 509 sentences for apples-to-apples cond-A comparison (RWTH sentences aren't in SiMT-660K, so no shipped GPT-4 chunks). Costs ~$5-20 API. Recommend doing after champion/scale-data results land.

**Unlocks:** appendix table in the paper. Reviewers ask for this — EAST has it.

## 10. Cross-annotator SFT (annotator transferability, NEW 2026-08-18)

Once cond-B annotations exist for all three backbones (E2B ✓, E4B in flight, Qwen in flight), build cond-B datasets from each and run off-diagonal SFTs:

| SFT backbone \ Annotator | E2B chunks | E4B chunks | Qwen chunks |
|---|---|---|---|
| E2B | ✓ done | queued | queued |
| E4B | queued | ✓ (self) | queued |
| Qwen | queued | queued | ✓ (self) |

6 off-diagonal runs. Each uses `--corpus_file results/phase2/condB_n10k_dataset_annotator-<X>.json` where the dataset is built via `scripts/phase2_build_condB_dataset.py --matrices results/phase2/annot_ot_condB_<X>_n10k/matrices.jsonl`.

**Hypothesis H10 (new).** If annotator chunk quality is model-invariant ("the annotator learned something universal about which positions are committable"), then off-diagonal cells should perform comparably to on-diagonal cells. If it's model-coupled ("annotator + SFT are a joint system"), off-diagonals should degrade.

**Prediction.** E4B-annotator → E2B-SFT should be within 1 BLEU of E2B-annotator → E2B-SFT (large model annotates a smaller model's training data well). Reverse (E2B-annotator → E4B-SFT) may show slight lift (smaller-model chunks generalize up).

**Unlocks:** paper's H5-descendant ablation. Directly answers "does the annotator's quality matter independently, or only via matched SFT?"

## 11. Multi-language pairs (paper stretch)

For ACL Findings / Main-track ambition, add at least one non-De→En language pair (Zh→En, Es→En recommended — WMT parallel corpora available). Full pipeline per pair: annotate (10K), SFT cond-A + cond-B, streaming eval. ~5 days per language on gpuhopper.

## Blockers, right now

- **matplotlib not installed** — copyq job 176531167 queued to fix. Blocks Figure 3 PDF generation but not the underlying numbers.
- **AL-CA measurement not implemented** — blocks apples-to-apples EAST Table 3 comparison. ~50 lines of code.
- **RWTH baseline decision** (GPT-4 API vs fast_align vs wait-k floor) — blocks Phase 3 appendix eval only.
- **No non-De→En language pair infra yet** — blocks multi-language stretch only.

## Not blockers (deferrable)

- Off-Multi-120K assembly (only if Stretch A pursued).
- Stage-II LoRA (only if Gate 3 passes at scale).
- BLEURT-20 fetch (only when moving past sacrebleu).
- Doc-level and conversational SiMT (Stretches B, C).

## Weekly checkpoint reminder

Bring `LOG.md` to Dipankar meetings, not a summary — HOUSEKEEPING §1.
