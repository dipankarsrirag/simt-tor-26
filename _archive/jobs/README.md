# jobs/_archive/

PBS wrappers from prior runs, kept for provenance. **Do not re-fire without supervisor approval** — many reference paths that have since moved.

## Grouping

### `v6b_gemma_2b/` — the current shipping baseline
All PBS files that produced the 171-cell eval matrix (`results/_archive/v6b_gemma_2b/extrinsic/`). Structured by pipeline stage:

- **`htgt_build/`** (1 file) — one-time source-pool build for the human-target corpus.
- **`htgt_annot/`** (16 files) — OT annotation across 8 directions × 2 batches (Aug 24).
- **`v2bal_v3_full/`** (40 files) — SFT training for the 3 v6b conditions (CondA, CondB, Ours).
- **`htgt_evals/`** (45 files) — extrinsic eval, primarily FLORES (deprecated post-contamination) and ar/vi.
- **`strategy_b_evals/`** (144 files) — main extrinsic eval: WMT15, WMT22, IWSLT17 de/ar × all 3 conditions.
- **`east8b_evals/`** (21 files) — EAST-8B baseline evals (Aug 28).
- **`condA_full/`** (20 files) — CondA baseline evals (FLORES; deprecated).
- **`v2bal_v3_wmt15/`** (5 files) — early WMT15 sanity subset (subsumed by `strategy_b_evals`).
- **`v2bal_full/`, `rb_fw_full/`** (~80 files) — pre-htgt v6b variants; abandoned in favour of v2bal_v3 + htgt.
- **`ctrl_envi_backfill/`** (4 files) — one-time env/vi backfill.
- **`download_east/`** (5 files) — one-time data-fetch for EAST corpus + Wmt17-21 human references.

### `legacy_loose/` — pre-organisation PBS soup
117 loose top-level PBS/sh files that predated the subdir organisation. Everything from `phase2_sft_multilingual_v5.pbs` (pre-v6 pivot) through `phase2_extrinsic_offline_*.pbs`. Never re-fire — most target paths that no longer exist.

Also includes `resubmit_missing_evals.sh` (the original) and the two `.list` files (`missing_evals.list`, `east8b_pending.list`) that fed the loop during the v6b eval sprint.

### `pbs_original/` — original `pbs/` directory contents
The old top-level `pbs/` directory (env.sh + templates) before consolidation into `jobs/`. Live copies now under `jobs/env.sh` and `jobs/templates/`.

### `deprecated_jobs_2026-08-21.tar.gz`
Tarball of an even earlier archive round (Aug 21) — pre-v6b pivot PBS wrappers.

## When to look here
- Reviewer asks "reproduce v6b evals" → `v6b_gemma_2b/{strategy_b_evals,east8b_evals}/*.pbs`. Path rewrite needed (`results/phase2/…` → `results/_archive/v6b_gemma_2b/…`).
- Reviewer asks for CondA baseline PBS → `v6b_gemma_2b/condA_full/` (targeted FLORES, so add a WMT-based re-eval PBS instead).
- Rebuilding from the loose-file era → search `legacy_loose/` by keyword; expect broken paths.
