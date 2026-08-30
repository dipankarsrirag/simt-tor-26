# Refactor plan — repo simplification for undergrad onboarding

Author: Dipankar Srirag · Draft: 2026-08-29 · **Status:** Awaiting sign-off. Nothing moves/deletes until user says go.

Goal: bring an undergrad assistant onto the codebase. Current state = 14 weeks of research crust: 48 scripts, 485 PBS files across 14 subdirs, 8 top-level markdown files, and a `tests/` directory with one stale test. Target state = a repo a new contributor can read end-to-end in an afternoon.

---

## Guiding principles

1. **One canonical way to do each pipeline step.** No `v5` vs `v6` vs `v6b` script forks in the live path — pick the winner, rename it to the plain name, archive the rest.
2. **Numbered scripts encode pipeline order.** `01_build_source_pool.py → 02_annotate.py → 03_build_sft_dataset.py → 04_train.py → 05_eval.py → 06_plot.py`. The undergrad reads them in order and understands the whole pipeline.
3. **Archive, don't delete.** Phase-1 exploration is evidence for Gate-1 decisions in `LOG.md`; deleting loses provenance. Move to `_archive/` subdirs with a `README.md` inside each explaining why they're there.
4. **Top-level docs = only what a new contributor needs.** Everything else lives under `docs/`.
5. **`LOG.md` stays at the top level.** It's the primary record — moving it would break every internal cross-reference.

---

## Current vs target tree

### Current (mess)

```
simt-tor-26/
├── CLAUDE.md               docs/experiments.md    docs/setup.md
├── LOG.md                  _archive/docs/method-formal.md         _archive/docs/OPTIONALS.md
├── docs/related-work.md         _archive/docs/TIMELINE.md       create-venv.sh
├── data -> …               docs/ (9 files)
├── figures/  jobs/ (14 subdirs, 485 PBS)  logs/  pbs/  results/  scripts/ (48)
├── src/ (annotator/ eval/ train/ constants.py + __pycache__/)
├── tests/ (1 file)
└── .claude/  .git/  .gitignore
```

### Target (clean)

```
simt-tor-26/
├── README.md               # NEW — one-page entry point. Claim, quickstart, links.
├── LOG.md                  # KEEP — append-only decision record.
├── create-venv.sh          # KEEP.
├── data -> …               # KEEP (symlink).
├── docs/
│   ├── 00-README.md        # Doc index (rewrite from current 00-README.md).
│   ├── 01-method.md        # ← _archive/docs/method-formal.md (moved from root).
│   ├── 02-hypotheses.md    # KEEP.
│   ├── 03-experiments.md   # ← docs/experiments.md (moved from root).
│   ├── 04-data.md          # KEEP (renamed from 06-data.md).
│   ├── 05-setup.md         # ← docs/setup.md §1-3 (trimmed).
│   ├── 06-related-work.md  # ← docs/related-work.md (moved).
│   ├── 07-next-steps.md    # KEEP.
│   ├── 08-followup-experiments.md   # KEEP (locked 2026-08-29).
│   ├── 09-refactor.md      # THIS FILE.
│   └── _archive/
│       ├── _archive/docs/OPTIONALS.md
│       ├── _archive/docs/TIMELINE.md
│       ├── CLAUDE-original.md      # (content folded into README + docs/01)
│       ├── HOUSEKEEPING-full.md    # (full ops manual)
│       ├── 03-phase1_annotator_experiments.md
│       ├── 04-random_floor_and_ot.md
│       └── 05-phase2_sft_and_streaming.md
├── src/
│   ├── __init__.py         constants.py
│   ├── annotator/          # annotate.py, criterion.py, east_format.py, boundary_refine.py
│   ├── train/
│   │   └── sft.py          # ← sft_v6.py (renamed). OLD sft.py → _archive/src/.
│   └── eval/
│       └── extrinsic.py    # KEEP.
├── scripts/
│   ├── 01_build_source_pool.py    # ← phase2_build_htgt_source_pool.py
│   ├── 02_build_sft_dataset.py    # ← phase2_build_sft_dataset.py
│   ├── 03_plot_bleu_al.py         # ← plot_v6b_bleu_al.py
│   ├── 04_score_comet.py          # ← phase2_score_comet.py
│   ├── prepare_tokenizer.py       # ← phase2_build_tokenizer_v6.py
│   ├── probe_east_8b_compat.py    # KEEP (used for baseline validation)
│   ├── make_job.py                # KEEP (PBS generator utility).
│   ├── download_data.sh           # KEEP.
│   ├── download_vi_en_test_sets.py # KEEP.
│   └── _archive/
│       ├── phase1_*.py (10 files)
│       ├── probe_*.py, smoke_*.py (10 files)
│       └── phase2_v[1-5]_*.py, phase2_verify_*.py, phase2_prep_*.py (~15 files)
├── jobs/
│   ├── annotate/                  # ← htgt_annot/  (16 PBS)
│   ├── train/                     # ← v2bal_v3_full/  (40 PBS)
│   ├── eval/
│   │   ├── gemma_2b/              # ← htgt_evals/ + strategy_b_evals/
│   │   └── east_8b/               # ← east8b_evals/
│   ├── loop_resubmit.sh           # ← resubmit_missing_evals.sh
│   ├── templates/                 # ← pbs/templates/
│   ├── env.sh                     # ← pbs/env.sh
│   ├── reproduce_v6b.sh           # NEW — end-to-end reproducer (calls 01→06 in order)
│   └── _archive/
│       ├── v2bal_full/            (40 PBS — pre-curated)
│       ├── rb_fw_full/            (40 PBS — abandoned rebucket-forward)
│       ├── condA_full/            (20 PBS — cond-A baseline, one-time reference)
│       ├── v2bal_v3_wmt15/        (5 PBS — subsumed by eval/gemma_2b)
│       ├── ctrl_envi_backfill/    (4 PBS — one-time backfill)
│       ├── download_east/         (5 PBS — one-time data fetch)
│       └── htgt_build/            (1 PBS — one-time source pool build)
├── results/
│   ├── phase2/                    # Current shipping models + extrinsic/ + datasets.
│   └── _archive/
│       └── phase1_*/              (11 dirs — Gate 1 evidence, no re-runs).
├── logs/                          # KEEP (regenerated).
├── figures/                       # KEEP (paper outputs).
└── .git/  .gitignore  .claude/
```

**Removed at top level:** `pbs/` (merged into `jobs/`), `tests/` (1 stale file → deleted, project doesn't have live unit tests). 6 markdown files (moved into `docs/`).

---

## Migration table

### Scripts (source → destination)

| Current | Destination | Reason |
|---|---|---|
| `scripts/01_build_source_pool.py` | `scripts/01_build_source_pool.py` | Live, entry stage. |
| `scripts/03_build_sft_dataset.py` | `scripts/03_build_sft_dataset.py` | Live. |
| `scripts/04_plot_bleu_al.py` | `scripts/04_plot_bleu_al.py` | Live (regenerated the 5 paper figures 2026-08-28). |
| `scripts/05_score_comet.py` | `scripts/05_score_comet.py` | Roadmap for COMET rescoring per `docs/08` §Metrics. |
| `scripts/prepare_tokenizer.py` | `scripts/prepare_tokenizer.py` | Utility; not numbered (one-time setup). |
| `scripts/probe_east_8b_compat.py` | (keep in place) | Used 2026-08-28 for EAST-8B baseline; live. |
| `scripts/make_job.py`, `download_*.py`, `download_data.sh` | (keep in place) | Utilities. |
| `scripts/phase1_*.py` (all 8) | `_archive/scripts/` | Gate 1 passed 2026-08-16; not re-runnable. |
| `scripts/probe_v6_*.py`, `probe_lookahead_*.py`, `probe_tau_sweep.py`, `smoke_load_gemma4.py`, `phase0_verify_east_format.py`, `phase2_streaming_smoke.py`, `phase2_inference_smoke.py`, `phase2_batched_ot_smoke.py`, `probe_annotator_batched.py`, `probe_annotator_kv_cache.py`, `probe_v6b_latency_diag.py` | `_archive/scripts/` | Diagnostics for closed bugs; historical only. |
| `scripts/phase2_build_condA_dataset.py`, `phase2_build_multilingual_source_pool.py`, `phase2_prep_indices.py`, `phase2_verify_loss.py`, `phase2_prepare_tokenizer.py`, `phase2_probe_*.py`, `phase2_compute_al_ca_approx.py`, `phase2_space_probe.py`, `phase2_plot_bleu_al.py`, `phase2_tau_sweep.py`, `plot_bleu_vs_al_all_conditions.py`, `compute_dal_from_stream.py` | `_archive/scripts/` | Pre-v6b variants; superseded. |

### src/ modules

| Current | Destination | Reason |
|---|---|---|
| `src/train/sft.py` | `_archive/src/sft_pre_v6.py` | Old base-model + latency-token recipe; deprecated 2026-08-21 (v6 pivot). |
| `src/train/sft.py` | `src/train/sft.py` | Rename to plain name — this is the live recipe. |
| `src/annotator/*`, `src/eval/extrinsic.py`, `src/constants.py` | (no change) | All live. |
| `src/**/__pycache__/` | delete | Runtime artifacts, in .gitignore. |

### Jobs directories

| Current | Destination | Reason |
|---|---|---|
| `_archive/jobs/gemma_2b_curated/htgt_annot/` | `jobs/annotate/` | Rename to pipeline-stage name. |
| `_archive/jobs/gemma_2b_curated/v2bal_v3_full/` | `jobs/train/` | Live training PBS. |
| `_archive/jobs/gemma_2b_curated/htgt_evals/` + `_archive/jobs/gemma_2b_curated/strategy_b_evals/` | `jobs/eval/gemma_2b/` | Merged (both target our 2B checkpoints). |
| `_archive/jobs/gemma_2b_curated/east8b_evals/` | `jobs/eval/east_8b/` | Live. |
| `jobs/loop_resubmit.sh` | `jobs/loop_resubmit.sh` | Rename to clearer intent. |
| `pbs/templates/`, `pbs/env.sh` | `jobs/templates/`, `jobs/env.sh` | Consolidate into `jobs/`. |
| `jobs/htgt_build/` | `_archive/jobs/htgt_build/` | 1 PBS, one-time build. |
| `jobs/v2bal_full/`, `rb_fw_full/`, `condA_full/`, `v2bal_v3_wmt15/`, `ctrl_envi_backfill/`, `download_east/` | `_archive/jobs/` | Superseded / one-time / abandoned. |
| `_archive/jobsd/` (existing dir with 1 tar.gz) | `_archive/jobs/` | Consolidate archive naming (`_archive` throughout). |

### Top-level markdown

| Current | Destination | Reason |
|---|---|---|
| `README.md` | **NEW** | Doesn't exist yet. Entry point for new contributor. |
| `CLAUDE.md` | `_archive/docs/CLAUDE-original.md` | Content folded into new README + `docs/01-method.md`. |
| `_archive/docs/method-formal.md` | `docs/01-method.md` | Detailed algorithm — belongs in docs. |
| `docs/experiments.md` | `docs/03-experiments.md` | Ablation grid; reference doc. |
| `docs/setup.md` | `docs/05-setup.md` (trimmed) + `_archive/docs/HOUSEKEEPING-full.md` | Trim to §1-3 (venv/paths/data fetch). Full 23KB version archived. |
| `docs/related-work.md` | `docs/06-related-work.md` | Lit review reference. |
| `_archive/docs/TIMELINE.md` | `_archive/docs/TIMELINE.md` | Superseded by LOG.md. |
| `_archive/docs/OPTIONALS.md` | `_archive/docs/OPTIONALS.md` | 74KB dumping ground; 80% is stale venue-targeting discussion. |
| `LOG.md` | (no change — top level) | Primary record. Internal cross-references. |

### Results directories

| Current | Destination | Reason |
|---|---|---|
| `results/phase1_*` (11 dirs) | `_archive/results/phase1_*` | Gate 1 evidence, no re-runs. |
| `_archive/results/gemma_2b_curated/sft_multilingual_v6b_v2bal_v3_htgt/` etc. | (no change) | Shipping models. |
| `_archive/results/gemma_2b_curated/extrinsic/` | (no change) | All 798 landed cells. |

### Delete outright

| Path | Reason |
|---|---|
| `tests/test_annotator_cpu_tiny.py` | 1 test, no CI, not maintained. User explicitly called out `tests/` as "random". |
| `tests/` (empty after) | Consolidate. |
| `**/__pycache__/` | Runtime artifacts; already in .gitignore. |

---

## New files to create

### `README.md` (top level)

Content sketch (~150 lines):

```markdown
# Teacher-Free Read/Write Annotation for Simultaneous Machine Translation

Undergrad research project. Supervisor: Dipankar Srirag (UNSW).

## What this project claims

[3-paragraph summary from CLAUDE.md §The claim + one-line result from LOG.md 2026-08-28]

## Repo tour

- `src/` — the library. Three subpackages: annotator (offline chunk-tag placement), train (SFT recipe), eval (streaming inference + metrics).
- `scripts/` — pipeline entry-points, numbered 01→06 in run order.
- `jobs/` — PBS wrappers for Gadi. `annotate/`, `train/`, `eval/`, plus `loop_resubmit.sh` for queue draining.
- `docs/` — everything else. Read `docs/README.md` first; it's the doc index.
- `LOG.md` — decision + run log. **Never edited retroactively.**

## Quickstart

1. `bash create-venv.sh` — sets up `/scratch/po67/ds9561/.venv-fil/`.
2. `source /scratch/po67/ds9561/.venv-fil/bin/activate` — activate.
3. See `docs/05-setup.md` for paths, HF cache, PBS access.

## Running the pipeline (v6b, shipping)

Full reproduction takes ~325 GPU-h on H200:

```
bash jobs/reproduce_v6b.sh                    # end-to-end
# or step by step:
python scripts/01_build_source_pool.py --config configs/curated.yaml
qsub jobs/annotate/annot_ot_multi_htgt_de-en.pbs   # × 8 directions
python scripts/03_build_sft_dataset.py --config configs/curated.yaml
qsub jobs/train/sft_multilingual_v6b_v2bal_v3_htgt.pbs
qsub jobs/eval/gemma_2b/wmt15_v2balv3htgt_medium.pbs   # × per-latency-per-dir
python scripts/04_plot_bleu_al.py
```

## Where to read next

- **Algorithm:** `docs/01-method.md`
- **Falsifiable claims:** `docs/hypotheses.md`
- **Experiment plan (paper submission):** `docs/08-followup-experiments.md`
- **Setup / paths / accounts:** `docs/05-setup.md`
```

### `jobs/reproduce_v6b.sh` (NEW — end-to-end wrapper)

Sequentially invokes 01→06 with the canonical config paths. Undergrad's "just make it run" button. Includes checkpoints so partial re-runs work.

### `docs/README.md` (rewrite)

New content: doc index with 3-sentence description of each 01→09 file. Points to `README.md` for the top-level onboarding.

### `_archive/scripts/README.md` (NEW inside archive)

One-liner per archived script explaining why it's there.

### Similarly for `_archive/jobs/README.md`, `_archive/results/README.md`, `_archive/src/README.md`, `_archive/docs/README.md`.

---

## Renaming discipline

- **Live scripts:** numbered prefix `NN_verb_noun.py`. Verbs: build, annotate, train, eval, plot, score, prepare, probe, download.
- **Live modules under `src/`:** plain name (no `_v6` suffix). Git preserves history.
- **Live PBS dirs:** pipeline-stage names (annotate, train, eval). Not backbone/dataset names.
- **Archive naming:** always `_archive/` (leading underscore, no dash, singular). Existing `_archive/jobsd/` renamed to match.

---

## Impact summary

| Change | Files affected |
|---|---|
| Scripts moved to `_archive/` | ~30 |
| Scripts renamed | 5 |
| Scripts kept in place | ~10 |
| PBS files moved | ~180 (across 8 subdirs into 3 live subdirs + `_archive/`) |
| Top-level MD moved to docs/ | 5 |
| Top-level MD archived | 2 (OPTIONALS, TIMELINE) |
| Top-level files remaining | 4 (README, LOG, create-venv.sh, .gitignore) |
| New files created | ~7 (README, reproduce_v6b.sh, 5× archive READMEs) |
| Deleted outright | `tests/`, all `__pycache__/` |

**Net effect:** top-level from 12 files to 5. `scripts/` from 48 to ~15 live. `jobs/` from 14 subdirs to 3 live + 1 `_archive/`. New contributor's entry cost: **README + 5 docs, ~20 min read** vs current 8 top-level MD files + 48 uncategorised scripts.

---

## Execution order (if approved)

**Phase A — safe (git-visible, reversible via revert):**
1. Create `_archive/docs/`, `_archive/scripts/`, `_archive/jobs/`, `_archive/results/`, `_archive/src/` (empty dirs with `.gitkeep`).
2. Move top-level MD files per table. Update all internal cross-references (`_archive/docs/method-formal.md` → `docs/01-method.md` etc.) via `grep -rl` + `sed`.
3. Move `phase1_*.py`, `probe_*.py`, `smoke_*.py`, `phase0_*`, `phase2_v[1-5]_*.py` to `_archive/scripts/`.
4. Rename active scripts to `01_..06_` prefix. Update any PBS files that reference them.
5. Move `src/train/sft.py` → `_archive/src/sft_pre_v6.py`. Rename `src/train/sft.py` → `src/train/sft.py`. Update `import` statements in `scripts/` and `jobs/`.
6. Consolidate `pbs/` into `jobs/`. Update `#PBS -o` paths and any `source $PBS_O_WORKDIR/pbs/env.sh` refs in PBS files.
7. Rename `_archive/jobs/gemma_2b_curated/htgt_annot/` → `jobs/annotate/` etc. Update PBS internal `#PBS -o /g/data/…/jobs/…/log` paths.
8. Move `jobs/v2bal_full/`, `rb_fw_full/`, etc. to `_archive/jobs/`.
9. Delete `tests/`, all `__pycache__/`.
10. Write `README.md`, `jobs/reproduce_v6b.sh`, all `_archive/README.md` files.
11. Commit as **one** atomic commit per phase (A.1-A.11) with clear messages.

**Phase B — destructive (one-way):**
- No hard deletes in phase A above beyond `tests/` and `__pycache__/`. Everything else moves to `_archive/`.
- If, after Phase A, you decide `_archive/` contents are surplus, `git rm -r` in a separate commit later.

**Estimated time:** 2-3 focused hours. Most of it is careful `sed`-across-PBS-files to keep paths in sync.

---

## Risks

| Risk | Mitigation |
|---|---|
| Broken PBS paths after `jobs/` restructure | Update all `#PBS -o` and `source` lines via scripted rewrite; smoke-test 1 PBS from each stage post-move. |
| Broken `import` after `src/train/sft.py` swap | `grep -rn "from src.train.sft"` before + after; verify identical count. |
| `LOG.md` cross-references break | LOG.md references `_archive/docs/method-formal.md`, `docs/experiments.md`, etc. — `sed -i 's|METHOD\.md|docs/01-method.md|g' LOG.md`. |
| `docs/*.md` cross-references break | Same treatment. |
| Undergrad thinks archived stuff is deleted | `_archive/README.md` explains what's there and when to look. |
| Refactor disrupts live jobs in queue | Complete refactor when queue is empty (currently is — verified 2026-08-29). |

---

## Questions for user before Phase A starts

1. **Delete `tests/` outright, or salvage `test_annotator_cpu_tiny.py` into a proper `tests/` dir with pytest wiring?** My default: delete. The test is stale (targets pre-v6 annotator), no CI runs it, keeping it invites the undergrad to trust a broken safety net.

Delete, the pipeline has been validated end to end wiht our experiments on gemma 4 2b

2. **`condA_full/` PBS — archive or keep live?** These are the baseline that produced the current CondA numbers in the paper. Explore agent said archive; I lean keep in `jobs/eval/gemma_2b/` since they're paper-referenced. **Default: keep live.**

what is this? explain properly

3. **Merge `CLAUDE.md` into README, or keep both?** CLAUDE.md is the current "read this first" doc for the AI. If you want the undergrad to also read it as-is, keep it. My default: merge (README = human onboarding, CLAUDE.md content lives inside it). If you keep, do we need CLAUDE.md to change now that content is duplicated?

When I push, anything CLAUDE related will need to be gitignored.

4. **Numbered script prefix (`01_`, `02_`) — yes or no?** Explicit pipeline order in filenames is undergrad-friendly but can look overly prescriptive. Alternative: keep plain names, document pipeline order in README only. **Default: yes, numbered.**

Yes.

5. **New `configs/` directory for YAML configs?** Currently every entry-point script has hardcoded paths (per Explore §Big findings #3). Adding `configs/curated.yaml` etc. is a genuine improvement for reproducibility but adds ~4 hours of work. **Default: defer — not in this refactor.**

Yes YAML configs and the scripts should stay.

Answer these 5 questions and I'll execute Phase A.
