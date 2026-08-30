# Teacher-Free Read/Write Annotation for Simultaneous Machine Translation

Undergrad research project. Supervisor: Dipankar Srirag (UNSW). Target venue: ACL/EMNLP Findings or IWSLT.

## The claim (one paragraph)

[EAST](https://aclanthology.org/2025.findings-acl.1045.pdf) (Findings of ACL 2025) teaches an LLM adaptive read/write behaviour by fine-tuning on data where **GPT-4** decided where the read/write tags go. We replace GPT-4 with the **backbone model's own predictive distributions**: for each parallel sentence pair, we hold the full source at data-construction time, measure when each target token's next-token distribution has converged to its full-source value, and place `<|end-of-read|>` there.

**Falsifiable claim:** backbone-derived tag placement matches or beats GPT-4-derived placement, with the margin growing on word-order-divergent pairs (e.g. German verb-final).

**Empirical status (2026-08-29):** Gemma-4-E2B-it self-annotated + fine-tuned dominates matched-backbone GPT-4 baseline on IWSLT17 (both de-en and en-de). Full 171-cell eval matrix in `_archive/results/v6b_gemma_2b/extrinsic/`. Follow-up experiments planned in `docs/followup-experiments.md`.

---

## Repo tour (top level)

```
├── README.md          ← you are here
├── LOG.md             append-only run + decision log — the primary record
├── create-venv.sh     bootstrap the Python environment
├── src/               the library (annotator, train, eval, config)
├── scripts/           Python entry-points, numbered 01→05 in pipeline order
├── bin/               shell launchers — run these; call scripts/*.py with the right env
├── configs/           YAML configs — one per experiment tag
├── jobs/              PBS wrappers for cluster batch (annotate/, train/, eval/)
├── results/           outputs (annotate/, sft_dataset/, train/, eval/)
├── logs/              PBS stdout/stderr (per-tag)
├── docs/              method, hypotheses, data, setup, refactor, etc.
├── _archive/          everything from prior runs (docs/, scripts/, jobs/, results/, src/, logs/)
├── data → …           symlink to the shared data root (paths documented in docs/setup.md)
└── figures/           paper output PNGs
```

Every live user-facing subtree (`jobs/`, `results/`, `logs/`) has three subdirs — `annotate/`, `train/`, `eval/` — for per-tag outputs. All prior runs live under `_archive/{jobs,results,logs}/v6b_gemma_2b/`. Everything a new experiment produces goes under `.../annotate/{annotator}/{pair}/`, `.../train/{tag}/`, `.../eval/{tag}/`. This keeps output namespaces separate across contributors and easy to collate.

---

## Prerequisites (any Linux/macOS box)

- **Python 3.10+** with pip
- **PyTorch with CUDA** for Stages 2, 4, 5 (annotation + SFT training + streaming eval). Stages 1, 3, 6 run on CPU.
- **~80GB disk** for the current baseline's cached models + training data (less if you fetch on demand).
- Optional: **PBS scheduler** (Gadi H200 cluster) — the `jobs/` directory has PBS wrappers; on any other GPU box just invoke `bin/*` directly.

Set two env vars before your first run (adjust for your filesystem):

```bash
export SIMT_MODEL_BASE=$HOME/models              # where HF backbones will live
export SIMT_DATA_ROOT=$HOME/data/simt-tor-26     # where parallel corpora live
```

Set them permanently in `~/.bashrc` (or `~/.zshrc`) so `bin/*` launchers pick them up automatically. `bin/_env.sh` will also fall back to sensible per-host defaults if you leave them unset (see §Environment below).

---

## Quickstart

```bash
git clone <this-repo>
cd simt-tor-26
bash create-venv.sh                    # sets up ./.venv (or reuses shared venv if present)
source .venv/bin/activate               # or your usual venv
pip install -r requirements.txt         # torch, transformers, datasets, sacrebleu, pyyaml

# Verify env
SIMT_ENV_VERBOSE=1 bin/03_build_sft_dataset --help   # will print resolved paths + arg help
```

Full setup notes (Gadi-specific paths, HF cache locations, `pyproject`/venv layering, PBS module list): **`docs/setup.md`**.

---

## Running an experiment

Every experiment is a YAML file under `configs/`. To run one end-to-end:

```bash
bin/run configs/{tag}.yaml --ngpus N
```

That's it. The runner reads the YAML, expands it into per-stage shell commands, sets `CUDA_VISIBLE_DEVICES=0..N-1`, and dispatches each stage in order.

Options:

| Flag | Purpose |
|---|---|
| `--ngpus N` | GPUs to use (default 1). Multi-GPU training uses `torchrun`. |
| `--stage 1..6` | Run only that stage. Default: all stages. |
| `--skip 1,3` | Skip stages (e.g. cross-annotation experiments skip Stage 2 — matrices already exist). |
| `--dry_run` | Print the commands without executing. Always do this first. |

## Pipeline (6 stages, tag-based)

For those curious what happens under the hood, or wanting to run a single stage manually:

**Two directories, one role each.** `scripts/*.py` = Python implementations. `bin/*` = shell launchers you run — they source `bin/_env.sh` for portable venv/cache handling and dispatch to `scripts/*.py`.

| Stage | Manual command | Output |
|---|---|---|
| 1. Build source pool (CPU) | `bin/01_build_source_pool --config configs/{tag}.yaml` | `results/sft_dataset/{tag}/source_pool.json` |
| 2. Annotate (**GPU required**) | `bin/02_annotate --input_json ... --model_path ... --output_dir ...` | `results/annotate/{annotator}/{pair}/matrices.jsonl` (keyed by annotator + pair — reusable across experiments) |
| 3. Build SFT dataset (CPU) | `bin/03_build_sft_dataset --matrices ... --corpus_json ... --output ...` | `results/sft_dataset/{tag}/sft_dataset.json` |
| 4. SFT training (**GPU × N**) | `torchrun --nproc_per_node=N src/train/sft.py --corpus_file ... --output_dir ...` | `results/train/{tag}/final/` + `sft_summary.json` |
| 5. Extrinsic eval (**GPU**) | `python src/eval/extrinsic.py --model_dir ... --tokenizer_dir ... --dev_src ... --dev_ref ... --latency ... --mode streaming` | `results/eval/{tag}/*.json` |
| 6. Plot (CPU) | `bin/04_plot_bleu_al` | `figures/{tag}/*.png` |

Stages 1, 3, and 6 need only CPU + a few GB RAM. Stages 2, 4, 5 need CUDA (any A100/H100/H200-class GPU; 24GB+ VRAM comfortably fits 2B backbones in bf16; 40GB+ for 4B; 80GB for 8B).

**Additional utilities** (all in `bin/`, no extension):
- `bin/05_score_comet --tag {tag}` — post-hoc COMET rescoring of eval JSONs.
- `bin/prepare_tokenizer --backbone {hf_id} --output {dir}` — extend a backbone's tokenizer with EOR/EOW special tokens (one-time per backbone).
- `bin/probe_east_8b_compat --model_dir {path}` — sanity-check a new backbone integrates with the pipeline.
- `bin/download_data`, `bin/download_vi_en_test_sets --out_dir {dir}` — one-time data fetches.
- `bin/make_job --config configs/{tag}.yaml --stage {annotate|train|eval}` — generate PBS wrappers (Gadi only).
- `jobs/loop_resubmit.sh` — queue-cap-aware batch resubmitter for large eval matrices (Gadi only).

---

## Environment

`bin/_env.sh` (sourced by every launcher) resolves paths in this priority order:

| Env var | Purpose | Fallback |
|---|---|---|
| `SIMT_REPO_ROOT` | this repo | auto-detected from `bin/_env.sh` location |
| `SIMT_VENV` | Python venv | Gadi shared venv → `./.venv` → `./.venv-fil` → system Python |
| `SIMT_HF_CACHE` | HuggingFace cache | Gadi shared cache → `$HOME/.cache/huggingface` |
| `SIMT_MODEL_BASE` | on-disk model weights root | Gadi shared cache → `$HOME/.cache/simt-models` |
| `SIMT_DATA_ROOT` | parallel corpora root | `$SIMT_REPO_ROOT/data` (symlinked) |
| `PYTHON` | python binary | `python3` |

**Verbose:** `SIMT_ENV_VERBOSE=1 bin/…` prints the resolved values on startup.

**YAML config env expansion:** placeholders like `${SIMT_MODEL_BASE}/gemma-4-E2B-it` in `configs/*.yaml` are expanded at load time (`src/config.load_config`).

---

## Running on Gadi (project ba39, NCI)

For collaborators with Gadi access — this project's original compute environment:

1. **Auth + storage:** join `ba39` and `po67` project groups. See `docs/setup.md` §1.
2. **venv:** shared at `/scratch/po67/ds9561/.venv-fil/`. `bin/_env.sh` picks it up automatically.
3. **Model + data caches:** shared at `/g/data/po67/dipankar/`. Same story.
4. **Submitting jobs:** `bin/make_job --config configs/{tag}.yaml --stage annotate` generates PBS files under `jobs/annotate/`. `qsub` them. Use `jobs/loop_resubmit.sh` for large matrices.
5. **Path convention:** every hardcoded absolute Gadi path in docs (like `/g/data/po67/dipankar/models`) is a **fallback** — override with your own `SIMT_*` env vars if you're on a different filesystem.

---

## Where to read next

- **Method:** `docs/method.md` — the annotator, mechanically.
- **Setup / paths / accounts (Gadi):** `docs/setup.md`.
- **What datasets, where they live, how to fetch:** `docs/data.md`.
- **Falsifiable claims:** `docs/hypotheses.md`.
- **Ablation grid + metrics:** `docs/experiments.md`.
- **Follow-up experiment plan (paper-facing):** `docs/followup-experiments.md`.
- **Refactor rationale (why the tree looks the way it does):** `docs/refactor.md`.
- **Prior art we build on:** `docs/related-work.md`.
- **Everything that happened, dated:** `LOG.md`.

## Non-negotiable invariants

1. **The test set is never annotated.** Annotation is training-data construction only.
2. **No reference or full-source signal reaches inference.** Streaming means streaming.
3. **Matched comparisons only.** Ours-tags vs GPT-4-tags must use the same sentences, same count, same backbone, same hyperparameters.
4. **Report AL-CA, not just AL.** Computation-aware latency is where policy methods die.
5. **Every experiment gets a `LOG.md` entry** with config, command, and result before the next one starts.
