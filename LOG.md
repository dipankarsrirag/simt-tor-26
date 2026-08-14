# Log

Append-only. Newest at the top. Two kinds of entry: **decisions** (what we chose and why) and **runs** (what we executed and what happened).

Log the run *before* starting the next one. A run without an entry did not happen.

---

## Template — decision

```
### [DECISION] YYYY-MM-DD — one-line summary
**Context:** what prompted this
**Options:** what was considered
**Chose:** what and why
**Revisit if:** the condition that would change this
```

## Template — run

```
### [RUN] YYYY-MM-DD — run-id
**Config:** backbone / data size / criterion / tau / seed
**Command:** exact invocation
**Result:** numbers, with the metric named
**Read:** what this means for the next step
```

---

<!-- entries below -->

### [SESSION HANDOFF] 2026-08-14 — end-of-session state

**Repo:** clean, on `main` at `9e120cb`, synced with `github.com/dipankarsrirag/simt-tor-26`.

**Docs written this session:** `CLAUDE.md` (dataset roles table + WMT test-set section), `METHOD.md`, `EXPERIMENTS.md` (Stage-I scope, WMT22 correction from Ar/Zh error), `TIMELINE.md` (Phase 0 concrete deliverables + Stretches A/B/C), `RELATEDWORKS.md` (two-stage recipe), `HOUSEKEEPING.md` (paths, compute, git, data table, venv discipline), `LOG.md` (this file), `OPTIONALS.md` (venue verdict, 3 blockers, 4 strengthening, 7 method improvements, closest-work distinctions, 2×2 novelty frame).

**Infrastructure scaffolded:** `.gitignore`, `create-venv.sh` (not yet run), `scripts/make_job.py` (gpuhopper+copyq only, shared `/g/data/po67/dipankar/cache/`), `pbs/env.sh`, `pbs/templates/job.pbs.tpl` (auto-resubmit), `src/constants.py`, `src/{annotator,train,eval}/`, `scripts/download_data.sh`, `data/` symlink to `/g/data/po67/dipankar/data/simt-tor-26/`.

**Pending — needs human decision before Phase 0 code starts:**

1. **Scale framing.** OPTIONALS.md §Blocker 1: Option A ("at 2B" preregistered) vs Option B (post-writeup 8B replication on `Llama-3.1-8B-Instruct`). Recommendation A. Blocks the paper's abstract wording; not blocking Phase 0 code.
2. **OPTIONALS.md method-improvement scope.** Which of M1–M7 go in the annotator. Recommendation: M1, M2, M3, M5, M7 (High-priority set + trivial M5). Blocks the annotator design — decide before Phase 1.
3. **Paper name.** Suggested `DRIFT` (Distributional Read/write Inference-Free Training). Not blocking code, but easier to fix before project-name strings enter scripts.

**Pending — infrastructure work not blocked on human decision:**

4. **RWTH De→En gold alignments URL.** `scripts/download_data.sh` step 5 is a TODO placeholder. EAST paper §E.4 has the source. Once URL is in, re-run `qsub jobs/download_data.pbs` (idempotent — will only fetch RWTH). Blocks the Gate 1 intrinsic annotation-quality measure.
5. **`bash create-venv.sh` — layers `pot / trl / accelerate / peft / datasets / sacrebleu` onto the shared `.venv-fil`.** Not yet run. Coordinate with `first-impressions-last` and `simul-mt` owners per HOUSEKEEPING §4.1 shared-venv discipline. Blocks any code that imports these packages.
6. **BLEURT-20 fetch to `MODEL_BASE/BLEURT-20/`.** Flagged in HOUSEKEEPING §5. Needed for the third-metric row in `EXPERIMENTS.md`. Trivial `copyq` job; not blocking early phases.
7. **`scripts/build_off_multi.py` — Off-Multi-120K assembly from WMT17-21 test data à la ALMA.** Only needed for Stretch A (multilingual Stage II), not for the primary Stage-I result.

**Context prime for next session.** Read order: `CLAUDE.md` (project spec + dataset table) → `OPTIONALS.md` (paper strategy; the 2×2 diagonal-move framing is the anchor) → `TIMELINE.md` Phase 0. Do not start writing the training pipeline — the annotator is the project, the SFT is plumbing.

---

### [RUN] 2026-08-14 — copyq download job 176225855.gadi-pbs
**Config:** copyq, 1 CPU / 8 GB / 100 GB jobfs, walltime 04:00:00. Job script `jobs/download_data.pbs` calls `scripts/download_data.sh`.
**Command:** `qsub jobs/download_data.pbs`
**Result:** `SiMT-De-En-660K` (660,876 rows, 685 MB — latency counts: low=230,902 / medium=227,131 / high=202,843), `SiMT-Multi-90K` (67 MB, 8 directions), WMT15 De-En newstest2015 (2,169 sentence pairs, 504 KB), WMT22 all 8 pairs `{de,en,zh,ru,cs}-{en,de,zh,ru,cs}` with `docid` (3.9 MB). RWTH and Off-Multi-120K skipped (TODOs). Log at `logs/download_data.log`.
**Read:** All Stage-I data assets are on disk at `/g/data/po67/dipankar/data/simt-tor-26/`. `data/` symlink from the repo resolves. Ready for Phase 0 format inspection and Phase 1 annotator development. RWTH still needed for Gate 1 intrinsic eval.

---

### [DECISION] 2026-08-14 — Scope: Stage I only; Stage II is stretch
**Context:** EAST is a two-stage recipe (§3.2 of the paper): full-weight SFT on `SiMT-De-En-660K` (Stage I, De→En) then LoRA on `SiMT-Multi-90K` + `Off-Multi-120K` (Stage II, 8 directions). Our 14-week timeline with a 2B backbone cannot cover both properly.
**Options:** (a) Stage I only, matched comparison at De→En. (b) Stage I + Stage II subset, sacrificing ablation depth. (c) Full recipe on a smaller data subset each — matches EAST shape but neither stage lands cleanly.
**Chose:** (a). The claim lives in the annotation criterion, which decides tag placement in Stage I; Stage II just LoRA-adds on top of Stage-I tags and can't move the criterion. EAST publishes Stage-I numbers separately (Figure 3 "EAST-Stage-I"), giving us a matched target. Stretches A, B, C in `TIMELINE.md` are the multilingual, document-level, and conversational extensions — all gated on Gate 3.
**Revisit if:** the Stage-I result lands early (say by week 8) with room to spare, and Dipankar wants to add multilingual before the writeup.

### [DECISION] 2026-08-14 — Primary backbone: `Qwen3.5-2B`
**Context:** EAST's Table 2 uses Llama-3-8B-Instruct. Our compute is one H200 per job (see `HOUSEKEEPING.md` §6), which comfortably fits 2B full-weight tuning with margin for the annotator's prefix-batch passes. Larger backbones would eat Phase 2 walltime that we need for `tau` sweeps and ablations.
**Options:** (a) `Qwen3.5-2B`, (b) `gemma-4-E2B-it`, (c) 4B variants of either.
**Chose:** (a) as primary, (b) as the cross-family annotator-ablation partner. Sizes matched at 2B so the annotator-model ablation isolates family, not scale. Scale-up to 4B stays available (both on disk) if Gate 3 passes with headroom.
**Revisit if:** `METHOD.md` §8 sanity checks show `Qwen3.5-2B` produces degenerate `i*[j]` traces (commit points cluster at sentence end). Then switch to `gemma-4-E2B-it` and re-check.

### [DECISION] YYYY-MM-DD — Annotator is the same model as the fine-tuning backbone
**Context:** EAST uses GPT-4 as an external annotator. We need to decide whether to self-annotate or use a larger teacher.
**Options:** (a) same model, (b) larger external annotator, (c) GPT-4 as in EAST.
**Chose:** (a). Cleaner claim — no external teacher, no distillation dependency, and tags are calibrated to the model that must act on them. A larger annotator would likely give better tags but reintroduces exactly the dependency we are criticising.
**Revisit if:** the cross-annotation ablation shows same-model annotation underperforms — that would mean error amplification dominates self-calibration.