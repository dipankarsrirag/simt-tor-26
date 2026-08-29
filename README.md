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

Every user-facing subtree (`jobs/`, `results/`, `logs/`) has three live subdirs — `annotate/`, `train/`, `eval/` — plus an `_archive/`. Prior runs live under `_archive/v6b_gemma_2b/`. Everything a new experiment produces goes under `.../annotate/{tag}/`, `.../train/{tag}/`, `.../eval/{tag}/` for that experiment's tag (e.g. `east_8b_htgt`, `gemma_4b_htgt`). This keeps output namespaces separate across contributors and easy to collate.

## Quickstart

```bash
bash create-venv.sh                                 # first time only
source /scratch/po67/ds9561/.venv-fil/bin/activate  # every session
```

Full setup, paths, HF cache, Gadi PBS conventions, and account onboarding: **`docs/setup.md`**.

## Pipeline (6 stages, tag-based)

Pick a tag (short lowercase-with-underscores, e.g. `east_8b_htgt`). Create `configs/{tag}.yaml` describing the run (see `configs/example.yaml`). Then run each stage:

| Stage | Command | Output |
|---|---|---|
| 1. Build source pool | `python scripts/01_build_source_pool.py --config configs/{tag}.yaml` | `results/sft_dataset/{tag}/source_pool.json` |
| 2. Annotate (OT chunk placement) | `qsub jobs/annotate/{tag}_<dir>.pbs` × per direction | `results/annotate/{tag}/matrices.jsonl` |
| 3. Build SFT dataset | `python scripts/02_build_sft_dataset.py --config configs/{tag}.yaml` | `results/sft_dataset/{tag}/sft_dataset.json` |
| 4. SFT training | `qsub jobs/train/{tag}.pbs` | `results/train/{tag}/final/` + `sft_summary.json` |
| 5. Extrinsic eval | `qsub jobs/eval/{tag}_<test>_<lat>_<dir>.pbs` (many) | `results/eval/{tag}/*.json` |
| 6. Plot | `python scripts/03_plot_bleu_al.py --config configs/plots.yaml` | `figures/{tag}/*.png` |

Additional utilities:
- `scripts/04_score_comet.py` — post-hoc COMET rescoring of eval JSONs.
- `scripts/prepare_tokenizer.py` — extend a backbone's tokenizer with EOR/EOW special tokens (one-time per backbone).
- `scripts/probe_east_8b_compat.py` — sanity-check a new backbone integrates with the pipeline.
- `jobs/loop_resubmit.sh` — queue-cap-aware batch resubmitter for large eval matrices.

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
