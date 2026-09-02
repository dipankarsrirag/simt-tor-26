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
| 01 | `01_llama_3_8b_curated.yaml` | Llama-3-8B-Instruct | curated | self | OT | Fig 1, 2, 3, 4 | annotations DONE; SFT queued |
| 02 | `02_llama_3_8b_machine_targets.yaml` | — | — | — | — | (was Fig 1/2 target-quality line) | **DEFERRED** 2026-09-01 — see LOG.md; successor is `Llama-3↺east(ot)` planned as `09_llama_3_east_ot.yaml` |
| 03 | `03_llama_3_8b_waitk.yaml` | Llama-3-8B-Instruct | curated | — | wait-k | Fig 2, 3 line 4 | SFT queued (dataset built, `scripts/07_waitk.py`) |
| 04 | `04_llama_3_8b_conv.yaml` | Llama-3-8B-Instruct | curated | awesome-align | conv-simt | Fig 2, 3 line 5 | scaffold (`scripts/07_conv.py`); awesome-align + mBERT installed (`177879495`); per-latency training strategy (a/b/c in LOG.md) still to pick |
| 05 | `05_gemma_4b_curated.yaml` | Gemma-4-E4B-it (4B) | curated | self | OT | Fig 4 middle | annotations DONE; SFT not yet queued |
| 06 | `06_gemma_4b_from_2b_annot.yaml` | Gemma-4-E4B-it | curated | Gemma-2B | OT | Fig 5 middle | ready (skip Stage 2) |
| 07 | `07_llama_3_8b_from_2b_annot.yaml` | Llama-3-8B-Instruct | curated | Gemma-2B | OT | Fig 5 right | ready (skip Stage 2) |
| 09 | `09_llama_3_east_ot.yaml` (unallocated) | Llama-3-8B-Instruct | east (N-matched, 4 dirs) | self | OT | (planned) | not written; ablation for chunker quality (OT vs GPT-4 semantic chunks) on identical source+target as released EAST-8B |

**2026-09-01 backbone switch + 2026-09-02 tag rename.** Configs 01–04, 07
now use `meta-llama/Meta-Llama-3-8B-Instruct` (base) instead of
`biaofu-xmu/EAST-8B` (already-EAST-trained). Rationale: fair chunk-quality
baseline — the wait-k / conv-simt arms cannot inherit EAST's prior
streaming exposure. **Tags renamed `east_8b_* → llama_3_8b_*`** on 2026-09-02
to prevent HF-repo naming confusion (`tor-simt-east-8b-curated` would've
contained a Llama-3 checkpoint). Plot labels (`EAST↺ours` etc.) retained
in each config's `plot:` section for figure-legend continuity. Released
`biaofu-xmu/EAST-8B` remains the published-competitor reference line only.
See `LOG.md`.

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
