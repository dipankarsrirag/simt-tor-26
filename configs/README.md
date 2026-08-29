# configs/

One YAML per experiment tag. Every pipeline stage reads its parameters from the tag's config, so the tag is a complete specification of a run.

## Naming

`{backbone}_{corpus}_{policy_suffix}.yaml` — lowercase, underscore-separated.

Examples matching the follow-up experiment plan (`docs/followup-experiments.md`):
- `v6b_gemma_2b.yaml` — the current baseline (Gemma-4-E2B-it + curated + OT).
- `east_8b_curated.yaml` — EAST-8B backbone, curated corpus, self-annotated (Fig 1, 2, 3, 4).
- `east_8b_east_matched.yaml` — EAST-8B backbone, east-corpus proportion-matched (Fig 2 line 3).
- `east_8b_waitk.yaml` — EAST-8B backbone, curated corpus, wait-k policy (Fig 2, 3 line 4).
- `east_8b_conv.yaml` — EAST-8B backbone, curated corpus, Conv-SiMT policy (Fig 2, 3 line 5).
- `gemma_4b_curated.yaml` — Gemma-4-E4B-it, curated, self-annotated (Fig 4 middle).
- `gemma_4b_from_2b_annot.yaml` — Gemma-4-E4B-it, curated, tags from Gemma-2B (Fig 5 middle).
- `east_8b_from_2b_annot.yaml` — EAST-8B, curated, tags from Gemma-2B (Fig 5 right).

## Structure

Every YAML has 7 top-level sections. See `example.yaml` for the canonical template.

- `tag` — the namespace (must match filename stem).
- `backbone` — model to fine-tune. `hf_id`, `local_path`, `is_instruct`, `tokenizer_dir`.
- `source_pool` — Stage 1. Corpus, per-direction row counts, target-quality filters.
- `annotate` — Stage 2. Annotator (self or cross), criterion (ot/js/waitk/conv/gpt4), τ, top-k, lookahead-k, latency bins.
- `sft_dataset` — Stage 3. Merge rules, collapsed-row policy.
- `train` — Stage 4. Epochs, batch size, LR, warmup, eval/save cadence, bf16, loss masking.
- `eval` — Stage 5. Test sets and per-direction sentence counts, latencies to run, policy (`check_argmax`/`wait_k`/...), mode.
- `plot` — Stage 6. Axes, ticks, colour, marker, legend label.

## Adding a new experiment

1. `cp configs/example.yaml configs/{your_tag}.yaml`.
2. Edit `tag`, `backbone`, `annotate.criterion`, etc.
3. Generate PBS wrappers: `python scripts/make_job.py --config configs/{your_tag}.yaml --stage {annotate,train,eval}` (produces PBS files under `jobs/{stage}/{tag}_*.pbs`).
4. Run the stages in order. Outputs land under `results/{stage}/{tag}/`, logs under `logs/{stage}/{tag}/`.
5. Add a `LOG.md` entry before starting the next experiment.
