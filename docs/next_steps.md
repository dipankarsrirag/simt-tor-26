# Next steps, in order

Ordered by priority. Each item states what it does, why now, and what unlocks after.

## 1. Extrinsic eval harness — Layer 1 offline BLEU on newstest2013 dev (in flight)

`src/eval/extrinsic.py --mode offline` scaffolded. Runs full-source greedy decode
under `<|latency|>` prompt, sacrebleu against the reference. First job:
`jobs/phase2_extrinsic_offline_dev.pbs` scores cond-A/n=10K and cond-B/n=2K on
newstest2013 (3000 sents). This is the "does the base translator work at all"
sanity — advisor threshold is offline BLEU > 10 before streaming is meaningful.

- Submit `jobs/phase2_extrinsic_offline_dev.pbs`; expect ~30 min wall (100-sent
  smoke inside the same job catches pipeline errors early).
- If both models > 10 BLEU: proceed to Layer 2 streaming.
- If either < 10: diagnose (tokenizer drift, checkpoint choice, latency-prompt
  mismatch) before building the state machine.

**Unlocks:** Layer 2 (streaming state machine, AL word units) and Layer 3
(AL-CA via `torch.cuda.Event`). Only after Layers 1–2 land on dev do we touch
newstest2015 (the test set — reported once, no re-tuning).

## 2. cond-B n=10K SFT (blocked on annotation)

Cond-B n=10K annotation is in flight (`jobs/phase2_annot_ot_condB_n10k_shard.pbs`,
job 176455997 → chained 176459737, ~76% done). Once `matrices.jsonl` reaches
9,567 rows + `DONE` marker:

- `python scripts/phase2_build_condB_dataset.py --tau 0.30 --matrices results/phase2/annot_ot_condB_n10k/matrices.jsonl --output results/phase2/condB_n10k_dataset.json`
- Submit cond-B n=10K SFT with the same recipe as cond-A n=10K (early-stopping,
  3 epoch cap, lr 2e-5, effective batch 16, val_frac 0.05).
- Verify + streaming smoke (`scripts/phase2_inference_smoke.py`).

**Unlocks:** the matched A-vs-B pair at n=10K — the row that carries the paper.

## 3. Matched A-vs-B extrinsic on newstest2013 dev

Once cond-B n=10K is trained AND Layer-2 streaming is working:
- Run both models under `--mode streaming` on newstest2013.
- Report BLEU + AL per condition, per latency prompt (low/med/high).
- Look for A-vs-B delta on: BLEU tie or B >, AL delta small.

**Unlocks:** newstest2015 (test) reporting.

## 4. Deferred (post-Gate-3)

- **Cross-backbone Qwen3.5-2B (H6).** Same recipe on 200 stratified indices, then n=10K.
- **Scale-up gemma-4-E4B (H7).** Only after n=10K result is defensible.
- **RWTH intrinsic A-score (Phase 3 appendix per 2026-08-16 decision).** Needs a
  baseline decision (GPT-4-API re-annotation of the 509 sents recommended).
- **Off-Multi-120K assembly** (Stretch A only).
- **Stage-II LoRA** (Stretch, only if Gate 3 passes).
- **BLEURT-20 fetch.** Add to Phase 2 metrics once COMET-DA baseline is on paper.
- **Doc-level and conversational SiMT** (Stretches B, C).

## Blockers, right now

- Layer-2 streaming design is documented in `src/eval/extrinsic.py`'s module
  docstring + the 2026-08-17 advisor spec in `LOG.md`. No design blockers.
- cond-B n=10K SFT is annotation-blocked; annotation ETA <4h from last check.

## Weekly checkpoint reminder

Bring `LOG.md` to Dipankar meetings, not a summary — HOUSEKEEPING §1.
