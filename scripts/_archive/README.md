# scripts/_archive/

Old scripts kept for provenance and reproducibility of prior gates. **Do not run without supervisor approval** — many hardcode paths that no longer exist, target the pre-v6 API, or produce results that are already superseded.

## Grouping

### `phase0_*` (Aug 14)
- `phase0_verify_east_format.py` — one-time smoke to confirm the EAST-format loader consumed shipped `source_chunks`/`target_chunks` correctly. Passed; not re-run.

### `phase1_*` (Aug 14–16) — Gate 1 diagnostics
Establish that OT-based commit points beat random / entropy-only / prefix baselines on the SiMT-660K signal test. Gate 1 passed 2026-08-16 (see `LOG.md`); these scripts are the evidence record.
- `phase1_tau_sweep.py`, `phase1_annotate_smoke.py` — τ-grid search on 48-sentence pilot.
- `phase1_entropy_sweep.py`, `phase1_random_floor.py` — signal-test baselines.
- `phase1_gpt4_pearson.py`, `phase1_precompute_gpt4_pearson.py`, `phase1_per_sentence_compare.py` — GPT-4-chunks as reordering-severity proxy.
- `phase1_reordering_bin.py` — stratified-by-reordering aggregate.
- `phase1_gate1_analyse.sh` — arbiter script.

### `probe_*`, `smoke_*` (Aug 15–24)
Closed-bug diagnostics. Each fixed a specific issue that has since shipped; kept in case a similar bug recurs.
- `probe_v6_directids.py`, `probe_v6_sanity.py`, `probe_v6_roundtrip.py` — training-vs-inference byte-level alignment (v6 pivot).
- `probe_lookahead_smoke.py`, `probe_lookahead_pilot.py` — abandoned latency-masking experiment.
- `probe_annotator_batched.py`, `probe_annotator_kv_cache.py` — annotator perf tuning.
- `probe_tau_sweep.py`, `probe_v6b_latency_diag.py` — one-off diagnostics.
- `smoke_load_gemma4.py` — model-load path sanity.
- `phase2_streaming_smoke.py`, `phase2_inference_smoke.py`, `phase2_batched_ot_smoke.py` — pipeline sanity at each v6b intermediate stage.

### `phase2_*` (Aug 16–24) — Superseded pre-v6b variants
Everything before the v6b + htgt pipeline froze. Consult `LOG.md` for context.
- `phase2_build_condA_dataset.py`, `phase2_build_multilingual_source_pool.py` — Cond-A / v5 dataset builders (superseded by v6b htgt).
- `phase2_prep_indices.py`, `phase2_verify_loss.py`, `phase2_prepare_tokenizer.py` — earlier tokenizer/index prep (superseded by `scripts/prepare_tokenizer.py`).
- `phase2_probe_multilang_ppl.py`, `phase2_probe_multilang_ppl_multibackbone.py`, `phase2_probe_v4_avg_interpolation.py`, `phase2_probe_v4_eot_diagnosis.py` — v4/v5 diagnostic ablations.
- `phase2_compute_al_ca_approx.py` — earlier AL-CA approximation (replaced by full AL-CA in `src/eval/extrinsic.py`).
- `phase2_space_probe.py` — embedding-space diagnostic (exploratory).
- `phase2_plot_bleu_al.py`, `plot_bleu_vs_al_all_conditions.py` — earlier plot drafts (replaced by `scripts/03_plot_bleu_al.py`).
- `compute_dal_from_stream.py` — delay-aware-latency approximation, unused.

## When to look here
- Reviewer asks "how did you validate criterion X" → `phase1_*.py`.
- New bug in annotation → `probe_annotator_*.py` scaffolding may help.
- Reproducing an old paper draft's numbers → `phase2_probe_*.py`.
