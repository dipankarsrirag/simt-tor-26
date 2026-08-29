# Teacher-Free Read/Write Annotation for Simultaneous Machine Translation

Undergrad research project. Supervisor: Dipankar Srirag (UNSW). Target venue: ACL/EMNLP Findings or IWSLT.

## The claim (one paragraph)

[EAST](https://aclanthology.org/2025.findings-acl.1045.pdf) (Findings of ACL 2025) teaches an LLM adaptive read/write behaviour by fine-tuning on data where **GPT-4** decided where the read/write tags go. We replace GPT-4 with the **backbone model's own predictive distributions**: for each parallel sentence pair, we hold the full source at data-construction time, measure when each target token's next-token distribution has converged to its full-source value, and place `<|end-of-read|>` there.

**Falsifiable claim:** backbone-derived tag placement matches or beats GPT-4-derived placement, with the margin growing on word-order-divergent pairs (e.g. German verb-final).

**Empirical status (as of 2026-08-29):** Gemma-4-E2B-it self-annotated + fine-tuned dominates matched-backbone GPT-4 baseline on IWSLT17 (both de-en and en-de). Full 171-cell eval matrix in `results/_archive/v6b_gemma_2b/extrinsic/`. Follow-up experiments planned in `docs/followup-experiments.md`.

## Repo tour (top level)

```
├── README.md          ← you are here
├── LOG.md             append-only run + decision log — the primary record
├── create-venv.sh     bootstrap the Python environment
├── src/               the library (annotator, train, eval)
├── scripts/           pipeline entry-points, numbered 01→04 in run order
├── configs/           YAML configs — one per experiment tag
├── jobs/              PBS wrappers for Gadi (annotate/, train/, eval/)
├── results/           outputs (annotate/, sft_dataset/, train/, eval/)
├── logs/              PBS stdout/stderr (per-tag)
├── docs/              method, hypotheses, data, setup, refactor, etc.
├── data → …           symlink to /g/data/po67/dipankar/data/simt-tor-26/
└── figures/           paper output PNGs
```

Every user-facing subtree (`jobs/`, `results/`, `logs/`) has three live subdirs — `annotate/`, `train/`, `eval/` — plus an `_archive/`. Prior runs live under `_archive/v6b_gemma_2b/`. Everything a new experiment produces goes under `.../annotate/{tag}/`, `.../train/{tag}/`, `.../eval/{tag}/` for that experiment's tag (e.g. `east_8b_curated`, `gemma_4b_curated`). This keeps output namespaces separate across contributors and easy to collate.

## Quickstart

```bash
bash create-venv.sh                                 # first time only
source /scratch/po67/ds9561/.venv-fil/bin/activate  # every session
```

Full setup, paths, HF cache, Gadi PBS conventions, and account onboarding: **`docs/setup.md`**.

## Pipeline (6 stages, tag-based)

Pick a tag (short lowercase-with-underscores, e.g. `east_8b_curated`). Create `configs/{tag}.yaml` describing the run (see `configs/example.yaml`). Then run each stage.

**Every stage has a `.py` (import-friendly) and a `.sh` (portable launcher) sibling.** Use the `.sh` on laptops or when you don't want to think about venv activation and HF cache paths — it auto-detects Gadi vs. laptop and does the right thing. `.py` is fine if you already have the venv active.

| Stage | On laptop / no venv | On Gadi (cluster) | Output |
|---|---|---|---|
| 1. Build source pool | `bash scripts/01_build_source_pool.sh --config configs/{tag}.yaml` | same (or `qsub` a PBS you generate) | `results/sft_dataset/{tag}/source_pool.json` |
| 2. Annotate (OT chunk placement) | `bash scripts/02_annotate.sh --config configs/{tag}.yaml --direction {pair}` | `qsub jobs/annotate/{tag}_<dir>.pbs` × per direction | `results/annotate/{tag}/matrices.jsonl` |
| 3. Build SFT dataset | `bash scripts/02_build_sft_dataset.sh --config configs/{tag}.yaml` | same | `results/sft_dataset/{tag}/sft_dataset.json` |
| 4. SFT training | (needs GPU — Gadi only) | `qsub jobs/train/{tag}.pbs` | `results/train/{tag}/final/` + `sft_summary.json` |
| 5. Extrinsic eval | (needs GPU — Gadi only) | `qsub jobs/eval/{tag}_<test>_<lat>_<dir>.pbs` (many) | `results/eval/{tag}/*.json` |
| 6. Plot | `bash scripts/03_plot_bleu_al.sh` | same | `figures/{tag}/*.png` |

Stages 1, 3, and 6 are laptop-runnable (no GPU); stages 2, 4, 5 need CUDA and are meant for Gadi (the .sh wrappers still work on any machine with the right compute).

Additional utilities:
- `bash scripts/04_score_comet.sh --tag {tag}` — post-hoc COMET rescoring of eval JSONs.
- `bash scripts/prepare_tokenizer.sh --backbone {hf_id}` — extend a backbone's tokenizer with EOR/EOW special tokens (one-time per backbone).
- `bash scripts/probe_east_8b_compat.sh --model_dir {path}` — sanity-check a new backbone integrates with the pipeline.
- `bash scripts/rebucket_latency.sh --input {file} --output {file}` — post-annotation latency-bin recomputation.
- `jobs/loop_resubmit.sh` — queue-cap-aware batch resubmitter for large eval matrices.

The shell wrappers source `scripts/_env.sh`, which:
- Auto-activates the venv (`/scratch/po67/ds9561/.venv-fil` on Gadi, `./.venv` on laptop, or falls back to system Python).
- Points HF cache at `/g/data/po67/dipankar/cache` on Gadi or `$HOME/.cache/huggingface` on laptop.
- Sets `PYTHONPATH` to the repo root.

Override any of these with `SIMT_VENV=/path`, `SIMT_HF_CACHE=/path`, `PYTHON=python3.11`, etc. Set `SIMT_ENV_VERBOSE=1` to log which paths got picked.

Reproduce the current headline (v6b Gemma-2B): follow the flow above with `tag = v6b_gemma_2b` — the completed run's artifacts already live under `results/_archive/v6b_gemma_2b/`.

## Where to read next

- **Method:** `docs/method.md` — the annotator, mechanically.
- **Setup / paths / accounts:** `docs/setup.md`.
- **What datasets, where they live, how to fetch:** `docs/data.md`.
- **Falsifiable claims:** `docs/hypotheses.md`.
- **Ablation grid + metrics:** `docs/experiments.md`.
- **Follow-up experiment plan (paper-facing):** `docs/followup-experiments.md`.
- **Refactor rationale:** `docs/refactor.md`.
- **Prior art we build on:** `docs/related-work.md`.
- **Everything that happened, dated:** `LOG.md`.

## Non-negotiable invariants

1. **The test set is never annotated.** Annotation is training-data construction only.
2. **No reference or full-source signal reaches inference.** Streaming means streaming.
3. **Matched comparisons only.** Ours-tags vs GPT-4-tags must use the same sentences, same count, same backbone, same hyperparameters.
4. **Report AL-CA, not just AL.** Computation-aware latency is where policy methods die.
5. **Every experiment gets a `LOG.md` entry** with config, command, and result before the next one starts.
