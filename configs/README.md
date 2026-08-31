# configs/

One YAML per experiment tag. Every pipeline stage reads its parameters from the tag's config. The `tag` field is the namespace for all outputs — `results/{stage}/{tag}/`, `logs/{stage}/{tag}/`, `figures/{tag}/`, `jobs/{stage}/{tag}_*.pbs`.

## Running an experiment

```bash
bin/run configs/{tag}.yaml [--ngpus N] [--stage 1..6|all] [--skip 1,2] [--dry_run]
```

The generic runner reads the YAML, expands the pipeline into concrete shell commands, sets `CUDA_VISIBLE_DEVICES=0..N-1`, and dispatches each stage. Use `--dry_run` first to inspect what will happen.

## Shipped configs (matches `docs/followup-experiments.md`)

Numeric prefix = run order (00 = baseline reproduction, then figure-by-figure through the followup plan).

| # | Config | Backbone | Corpus | Annotator | Policy | Figures | Status |
|---|---|---|---|---|---|---|---|
| 00 | `00_gemma_2b_curated.yaml` | Gemma-4-E2B-it (2B) | curated | self | OT | baseline (already run) | ✓ |
| 01 | `01_east_8b_curated.yaml` | EAST-8B (Llama-3-8B) | curated | self | OT | Fig 1, 2, 3, 4 | ready |
| 02 | `02_east_8b_east_matched.yaml` | EAST-8B | east (proportion-matched) | self | OT | Fig 1, 2 line 3 (Q1b) | ready |
| 03 | `03_east_8b_waitk.yaml` | EAST-8B | curated | — | wait-k | Fig 2, 3 line 4 | ready (`scripts/07_waitk.py`; smoke-passed 2026-08-31) |
| 04 | `04_east_8b_conv.yaml` | EAST-8B | curated | awesome-align | conv-simt | Fig 2, 3 line 5 | scaffold only (`scripts/07_conv.py`); awesome-align + mBERT install queued as `177879177`; Wang 2024 §2.1 chunk-assignment rule still to pick |
| 05 | `05_gemma_4b_curated.yaml` | Gemma-4-E4B-it (4B) | curated | self | OT | Fig 4 middle | ready |
| 06 | `06_gemma_4b_from_2b_annot.yaml` | Gemma-4-E4B-it | curated | Gemma-2B | OT | Fig 5 middle | ready (skip Stage 2) |
| 07 | `07_east_8b_from_2b_annot.yaml` | EAST-8B | curated | Gemma-2B | OT | Fig 5 right | ready (skip Stage 2) |

Cross-annotation configs (`06_*`, `07_*`) reuse the existing
`results/annotate/gemma-4-E2B-it/{pair}/matrices.jsonl` files — no re-annotation needed. Run with `--skip 2`.

## YAML structure

7 top-level sections, see `example.yaml` for the canonical template.

- `tag` — the namespace (must match filename stem).
- `backbone` — `hf_id`, `local_path`, `is_instruct`, `tokenizer_dir`.
- `source_pool` — Stage 1. Corpus (`curated` | `east`), per-direction row counts, target-quality filters.
- `annotate` — Stage 2. Annotator (`same_as_backbone` | HF id | local path), criterion (`ot` | `js` | `wait-k` | `conv-simt`), τ, top-k, lookahead-k, latency bins.
- `sft_dataset` — Stage 3. Merge rules, collapsed-row policy.
- `train` — Stage 4. Epochs, batch, LR, warmup, cadence, bf16, loss masking.
- `eval` — Stage 5. Test sets → per-direction sentence counts, latencies to run, policy, mode.
- `plot` — Stage 6. Colour, marker, legend label.

Env var placeholders `${SIMT_MODEL_BASE}`, `${SIMT_REPO_ROOT}`, etc. are expanded at load time by `src/config.load_config`.

## Adding a new experiment

1. `cp configs/example.yaml configs/NN_{your_tag}.yaml`  (next free `NN`)
2. Edit `tag` (must equal the filename stem without the number, or keep the full `NN_tag` — the runner uses whatever `tag:` says as the namespace).
3. Edit `backbone`, `source_pool.corpus`, `annotate.*`, and any per-run hyperparameters.
4. Dry-run: `bin/run configs/NN_{your_tag}.yaml --dry_run`.
5. If it looks right: `bin/run configs/NN_{your_tag}.yaml --ngpus N`.
6. Add a `LOG.md` entry before starting the next experiment.
