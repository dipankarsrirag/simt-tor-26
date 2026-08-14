# HOUSEKEEPING.md — simt-tor-26

Compute, paths, accounts, admin. Standing operational rules for the
Teacher-Free Read/Write Annotation project (`simt-tor-26`). No scientific
claims here — see `CLAUDE.md`, `METHOD.md`, `EXPERIMENTS.md`.

**Maintained by Dipankar.** If a rule below blocks you, message me before
working around it. This is a taste-of-research project (14 weeks) — the
housekeeping exists so you spend the time on the annotator, not on
re-discovering why a job dies on `gdata` quota.

Paths assumed throughout:

| Purpose | Path |
|---|---|
| Working dir | `/g/data/ba39/dipankar/simt-tor-26/` |
| PBS project | `po67` (compute) — `ba39` is storage only |
| Storage flags | `-l storage=gdata/ba39+gdata/po67` on every job |
| venv (compute) | `/scratch/po67/ds9561/.venv-fil/` — **shared with `first-impressions-last` and `simul-mt`**; layer extras via `bash create-venv.sh` (see §4) |
| venv (scoring) | `/g/data/po67/dipankar/venvs/comet/` — shared COMET env; call its `bin/python` directly, do not activate on top of `.venv-fil` |
| uv binary | `/g/data/po67/dipankar/uv/bin/uv` (do all installs through this) |
| Model weights | `/g/data/po67/dipankar/models/` (referred to as `MODEL_BASE`) |
| HF / torch / triton cache root | `/g/data/po67/dipankar/cache/` — **shared cache on po67 gdata**, same as `../arabic-dial-mt/pbs/env.sh`. Not scratch, not `$HOME`. |
| Datasets | `/g/data/po67/dipankar/data/simt-tor-26/` (symlinked into `./data`) |
| Logs | `/g/data/ba39/dipankar/simt-tor-26/logs/` (gitignored) |
| Jobs | `/g/data/ba39/dipankar/simt-tor-26/jobs/` (committed) |

---

## 1. Accounts and access

- **NCI Gadi.** You will use my `ba39` / `po67` allocation — you do not
  need a personal NCI project. Log in with your own NCI username, then
  charge jobs to `-P po67` and add storage flags for `ba39` and `po67`.
- **HuggingFace token.** Read-only token in `~/.netrc` on the login node.
  Never inline it in code, never commit `.env` files, never reproduce
  the plaintext token found in `/g/data/po67/dipankar/models/get_model.py`.
  Use `$HF_TOKEN` at runtime if you must.
- **Weekly meeting.** 30 minutes, in person or Zoom. Bring `LOG.md`, not a
  summary. If a gate in `TIMELINE.md` fails, message me the same day —
  do not wait for the next standing meeting.
- **Escalate before you spend.** Any single job over ~200 SU (roughly a
  full night on 4×H200), or any download over ~50 GB, ping me first. SU
  is real and shared with three other projects on `po67`.

---

## 2. Repository layout

```
simt-tor-26/
├── CLAUDE.md              # project spec — the claim, invariants, docs map
├── METHOD.md              # the annotation algorithm, precisely
├── EXPERIMENTS.md         # ablation grid, baselines, metrics
├── TIMELINE.md            # phases, gates, weeks
├── RELATEDWORKS.md        # what exists, what we build on
├── LOG.md                 # decisions + runs, append-only
├── HOUSEKEEPING.md        # this file
├── create-venv.sh         # layers simt-tor-26 extras onto shared .venv-fil (see §4)
├── .venv-freeze.txt       # full uv pip freeze of .venv-fil after last layering
├── .gitignore
├── data/       -> symlink to /g/data/po67/dipankar/data/simt-tor-26/ (never committed)
├── src/                   # library code + annotator + runners
│   ├── annotator/         # METHOD.md §§1–4 lives here
│   ├── train/             # SFT wrapper (trl SFTTrainer)
│   └── eval/              # intrinsic (RWTH) + extrinsic (SimulEval) harnesses
├── scripts/               # one-shot utilities incl. make_job.py
├── jobs/                  # generated PBS scripts (committed)
├── logs/                  # PBS stdout/stderr + metrics.jsonl (gitignored)
├── results/               # summaries + per-run JSONs (committed if <10 MB)
├── docs/                  # design notes, dataset notes
└── tests/                 # smoke tests for the annotator + SFT dataset builder
```

Every runner in `src/` accepts, at minimum: `--seed` (default 42),
`--output_dir` (absolute), `--config` (YAML). Results go to `results/`;
raw predictions go to `results/<run>/predictions.jsonl`.

---

## 3. Data

All EAST datasets and test sets, none of them live in the repo. Fetched
by `scripts/download_data.sh` running on `copyq`.

| Asset | Path under `data/` | Role (EAST paper) | Fetch |
|---|---|---|---|
| `SiMT-De-En-660K` | `SiMT-De-En-660K/` | **Stage I SFT** (De→En, GPT-4 chunks at 3 latency levels, WMT15-derived) | `hf download biaofu-xmu/SiMT-De-En-660K` |
| `SiMT-Multi-90K` | `SiMT-Multi-90K/` | **Stage II LoRA** (8 directions De/Zh/Ru/Cs↔En, GPT-4 chunks) — stretch only | `hf download biaofu-xmu/SiMT-Multi-90K` |
| `Off-Multi-120K` | `off-multi-120k/` | **Stage II LoRA** OMT co-training (WMT17-21 test data à la ALMA) — stretch only, **not on HF**, assembly script TODO | `scripts/build_off_multi.py` (unwritten) |
| `WMT15 De→En` newstest2015 | `wmt15-de-en/` | Primary SiMT test (EAST Figure 3) | `sacrebleu -t wmt15 -l de-en` |
| `WMT22 X↔En` 8 pairs | `wmt22/<pair>/` | Multilingual + document-level SiMT test (EAST Figure 4 + §4.3, stretch) | `sacrebleu -t wmt22 -l <pair>` |
| RWTH De→En gold alignments | `rwth-de-en/` | Intrinsic annotation-quality eval (EAST Appendix E.4). URL TODO | curl → tar |

Rules:

- `data/` is a **symlink** to `/g/data/po67/dipankar/data/simt-tor-26/`.
  Never a real directory in the repo. Never `git add data/`.
- Downloads live behind `scripts/download_data.sh`. The script is
  idempotent — already-present datasets are skipped. Add new fetches to
  the script, not ad-hoc.
- Do not re-download `SiMT-De-En-660K` if you already have it (~700 MB).
  Check `du -sh data/SiMT-De-En-660K` first.
- **Do not touch any test set during development.** WMT15 De→En is the
  primary test set; WMT22 pairs are stretch test sets. Threshold
  selection (`tau`) is done on WMT dev only. `EXPERIMENTS.md`
  §Guardrails is not a suggestion.
- Any new dataset requires a `docs/data/<name>.md` note: source URL,
  date fetched, checksum, license, and what preprocessing was applied.

---

## 4. Environment and dependencies

**Two shared venvs, same as `../simul-mt/` and `../first-impressions-last/`.**
Do not create a project-private venv. Layering onto the shared ones
keeps our per-user inode footprint on `po67` scratch bounded (a second
full torch venv exceeds the quota).

| Venv | Path | Owns | We add |
|---|---|---|---|
| `.venv-fil` | `/scratch/po67/ds9561/.venv-fil/` | first-impressions-last owns the base: torch, transformers, tokenizers, vllm, huggingface_hub, sentencepiece, numpy, safetensors, flashinfer. | Annotator: `pot` (Sinkhorn). SFT: `trl`, `accelerate`, `peft`, `datasets`. Eval: `sacrebleu`. |
| `venvs/comet` | `/g/data/po67/dipankar/venvs/comet/` | Shared scoring env: `unbabel-comet` and its pinned torch. | Nothing — use as-is for COMET-DA / COMET-Kiwi. Add BLEURT here (not in `.venv-fil`) once fetched. |

Activation:

```bash
source /scratch/po67/ds9561/.venv-fil/bin/activate            # annotator, SFT
source /g/data/po67/dipankar/venvs/comet/bin/activate         # scoring
```

The `offline_qwen3_4b.pbs` job in `../simul-mt/jobs/` shows the two-env
pattern in a single script: run the model with `.venv-fil`, then invoke
`/g/data/po67/dipankar/venvs/comet/bin/python` directly for the COMET
pass. Copy that pattern.

### 4.1 Adding a dependency (shared-venv discipline)

Only **additive** installs are automatically safe. If simt-tor-26 needs
a version *bump* on a package already installed by fil (torch,
transformers, vllm, flashinfer, ...), stop — that change affects
first-impressions-last and simul-mt too. Coordinate in `../first-impressions-last/`
and `../simul-mt/` before running the bump.

Our extras go into a **`create-venv.sh`** at repo root, ported from
`../simul-mt/create-venv.sh` — same shape, different package list. Add
new deps to that script and re-run; do not `pip install` ad-hoc. The
script uses uv:

```bash
/g/data/po67/dipankar/uv/bin/uv pip install \
    --python /scratch/po67/ds9561/.venv-fil/bin/python \
    pot trl accelerate peft datasets sacrebleu
```

Training stack is **PyTorch + Hugging Face `transformers` + `trl`**
(`SFTTrainer` for the annotated read/write data). Not LLaMA-Factory
— the trl route stays closer to the base fil venv, needs fewer extras,
and gives us a straight `Trainer` API for the annotator's teacher-forced
logit passes too.

### 4.2 After a scratch purge

Scratch purges happen every ~90 days on `po67`. When `.venv-fil`
vanishes:

1. Rebuild the base env from `../first-impressions-last/` first (they
   own it).
2. Run `bash create-venv.sh` here to layer simt-tor-26's extras.
3. Run `bash ../simul-mt/create-venv.sh` too, so simul-mt keeps
   working.

A dep that only exists on the live venv but not in `create-venv.sh` is
a dep that will disappear after the next purge. If you add a package,
add it to the script in the same commit.

### 4.3 uv

All installs go through `uv`, not raw `pip`. Binary:
`/g/data/po67/dipankar/uv/bin/uv`. See `../UV.md` for the group recipe
and known gotchas (setuptools pin, scispaCy-style URL wheels). uv
resolves and installs an order of magnitude faster than pip and is what
`create-venv.sh` uses.

### 4.4 Reproducibility freeze

Alongside `create-venv.sh` (which lists our extras), commit a
`.venv-freeze.txt` produced by
`uv pip freeze --python /scratch/po67/ds9561/.venv-fil/bin/python`
after any layering change. Same pattern as `../simul-mt/.venv-freeze.uv.txt`
in the parent dir. The freeze captures the full solved graph including
first-impressions-last's base — that is what a reviewer or a rebuild
after purge actually needs.

---

## 5. Models

**Backbone plan.** We are *not* reproducing EAST at Llama-3-8B. The
released `SiMT-De-En-660K` ships GPT-4 tags already, so both matched
conditions (A = GPT-4 tags, B = our tags) can run on any backbone we
choose. EAST's published numbers become a *sanity reference* only, not
a matched comparison — consistent with `EXPERIMENTS.md`
§Baselines ("Reproducing EAST's exact curve is a sanity check, not a
deliverable"). This drops the 8×A100 compute floor entirely.

**Start small — 2B primary.** The 14-week timeline and the
annotator-first plan (`TIMELINE.md` Phase 1) mean we want fast
iteration on the annotation criterion, not maximum absolute BLEU.
2B is fast enough to sweep `tau`, run the `METHOD.md` §8 sanity
checks daily, and re-annotate at will.

Ladder on disk (`MODEL_BASE = /g/data/po67/dipankar/models/`):

| Family | Model | Path | Role |
|---|---|---|---|
| Qwen-3.5 | `Qwen3.5-2B` | `Qwen3.5-2B/` | **Primary backbone** for both annotator and SFT (per `METHOD.md` §5). |
| Gemma-4 | `gemma-4-E2B-it` | `gemma-4-E2B-it/` | **Cross-family replication** at matched size — annotator-model ablation (`EXPERIMENTS.md` §Ablation grid, row 2). |
| Qwen-3.5 | `Qwen3.5-4B` | `Qwen3.5-4B/` | Scale-up candidate only after Gate 3 in `TIMELINE.md` passes. |
| Gemma-4 | `gemma-4-E4B-it` | `gemma-4-E4B-it/` | Second scale-up candidate; keep parity with the 2B ablation. |

Larger variants (`Qwen3.5-9B`, `Qwen3.5-27B`, `gemma-4-31B-it`) exist
on disk but are out of scope. Do not scale beyond 4B in this project.

Scorers on disk:

- `wmt22-comet-da/` — COMET-DA reference scorer.
- `wmt22-cometkiwi-da/` — reference-free COMET-Kiwi.

Not on disk (fetch on `copyq` when you get to `EXPERIMENTS.md` metrics):

- `lucadiliello/BLEURT-20` — needed for the BLEURT number reported
  alongside BLEU and COMET. Fetch to `MODEL_BASE/BLEURT-20/`.

Rules:

- **Same model for annotation and fine-tuning** per `METHOD.md` §5.
  Annotate with `Qwen3.5-2B` → fine-tune `Qwen3.5-2B`. The annotator-model
  ablation is a separate, deliberate axis (`gemma-4-E2B-it` annotates →
  `gemma-4-E2B-it` fine-tunes, then compare).
- **Never download at experiment runtime.** GPU nodes have no internet;
  the job will fail after burning walltime on DNS timeouts. All pulls go
  through `copyq` via `hf download --local-dir MODEL_BASE/<name>/`.
- Reference models by `MODEL_BASE + relative path` in code. Hard-code
  the absolute prefix in exactly one place — a `src/constants.py`.

---

## 6. PBS jobs on GADI

**Two queues only.** `gpuhopper` for compute, `copyq` for downloads.
Every other queue (`dgxa100`, `gpuvolta`, `normal`) exists but we do
not touch it — same convention as `../simul-mt/` and `../arabic-dial-mt/`.
Sticking to one GPU queue keeps results comparable (same silicon, same
kernel dispatch) and skips a class of "worked on V100, OOMs on A100"
bugs.

### 6.1 Never hand-write PBS

All PBS scripts under `jobs/` are generated by `scripts/make_job.py`
(port from `../simul-mt/scripts/make_job.py` on first use; adjust
`WORK_DIR` and `VENV`). Hand-editing the header silently gets
`ncpus`/`mem`/`storage` wrong.

Invocation:

```bash
python scripts/make_job.py \
    --name  <job_name> \
    --queue <gpuhopper|copyq> \
    --ngpus <N>                       # ignored for copyq
    --walltime HH:MM:SS \
    --script "python src/annotator/annotate.py --config configs/annot_smoke.yaml" \
    --output jobs/<job_name>.pbs
```

### 6.2 Canonical gpuhopper header

The generator writes this exact shape (per-GPU resource lines scale
with `--ngpus`; 4B backbones fit on `ngpus=1`):

```bash
#!/bin/bash
#PBS -N <job_name>
#PBS -P po67
#PBS -q gpuhopper
#PBS -l ncpus=12                     # 12 per GPU on gpuhopper — do not override
#PBS -l ngpus=1
#PBS -l mem=240GB                    # 240 GB per GPU
#PBS -l jobfs=120GB                  # 120 GB per GPU
#PBS -l walltime=HH:MM:SS
#PBS -l storage=gdata/ba39+gdata/po67
#PBS -l wd
#PBS -j oe
#PBS -k oed                          # write .o/.e directly, no staging
#PBS -o /g/data/ba39/dipankar/simt-tor-26/logs/<job_name>.log
# Queue: gpuhopper | GPU: H200-141GB x<N>
```

Non-negotiables:

- `-P po67` (compute charged here) and `-l storage=gdata/ba39+gdata/po67`
  (drop `ba39` and repo reads fail; drop `po67` and model reads fail).
- `-o` is an **absolute path** to `logs/`. Relative `-o` silently drops
  to `$HOME`.
- Never override `ncpus`/`mem`/`jobfs` — gpuhopper enforces
  `ncpus == 12 × ngpus`. If a job genuinely needs more RAM than the
  per-GPU allocation gives, request more GPUs; do not hand-edit.
- Never span nodes — max GPUs per job = max GPUs per node.

### 6.3 Baked environment (every gpuhopper job)

- **Redirect every cache to `/g/data/po67/dipankar/cache/`** — the
  shared cache on po67 gdata, same as `../arabic-dial-mt/pbs/env.sh`.
  `$HOME` has a tiny quota (a single HF shard blows it); scratch gets
  purged every ~90 days. The po67 cache persists and is deliberately
  shared across projects so we don't re-download the same weights three
  times. Export these all:
  `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`,
  `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`,
  `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TMPDIR`.
  All point under `/g/data/po67/dipankar/cache/*`.
- Offline flags for the HF stack: `HF_HUB_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. gpuhopper has no
  internet — a live lookup will hang until walltime kills it.
- `cd /g/data/ba39/dipankar/simt-tor-26` explicitly; do not rely on
  `-l wd` alone.
- `module load python3/3.10.4` before
  `source /scratch/po67/ds9561/.venv-fil/bin/activate` (fil's base env
  was built against 3.10.4 — matches `../simul-mt/jobs/`).
- For the scoring pass, do not re-activate — invoke
  `/g/data/po67/dipankar/venvs/comet/bin/python` directly (same as
  `../simul-mt/jobs/offline_qwen3_4b.pbs`).

### 6.4 Walltime discipline

- **Walltime ceiling on gpuhopper is 48h.** Break long jobs at
  checkpoint boundaries; rely on the auto-resubmit contract (§6.5).
- Pad your estimate by ~30% over measured runtime. Over-padding wastes
  queue priority; under-padding loses the whole run. If you don't know,
  smoke on `ngpus=1 walltime=00:15:00` first.
- Prefer several short jobs over one long one — kinder to fairshare
  and easier to restart on failure.

### 6.5 copyq — downloads only

- **The only queue with internet.** All HF pulls, WMT downloads, and
  wandb sync happen here. GPU jobs must be able to run fully offline.
- 1 CPU / 8 GB / 100 GB jobfs. 10h walltime ceiling.
- Never request a GPU on copyq; never request internet on gpuhopper
  (there is none).
- Template for HF pulls:
  `hf download <repo> --local-dir /g/data/po67/dipankar/models/<name>/`.
  Verify size against the HF repo page after — a truncated shard causes
  silent load failures.

### 6.6 Auto-resubmit contract (SFT runs)

Because SFT will bump the 48h ceiling on 660K, every training job
writes exactly one marker under `output_dir/` on exit:

- `DONE` — target steps reached.
- `NEEDS_RESUME` — `WalltimeGuardianCallback` stopped it before wall time.
- `FAILED` — died without a marker (wrapper writes it via `atexit`).

The PBS wrapper resubmits itself on `NEEDS_RESUME` up to a cap of **10**.
`FAILED` is terminal; only a human clears it after diagnosis. Do **not**
write code that wipes `FAILED`. Port
`../arabic-dial-mt/pbs/templates/job.pbs.tpl` — it already implements
this contract.

### 6.7 Submission and monitoring

- `qsub jobs/<name>.pbs` from the repo root; note the job ID.
- `qstat -u $USER` for your queue; `qstat -f <jobid>` for full status;
  `qdel <jobid>` to cancel. Kill runaway jobs the moment you notice —
  they burn real SU.
- Every job writes to `logs/<name>.log`. `tail -f logs/<name>.log`
  while it runs.
- Reuse job names carefully — resubmit overwrites the log. If you need
  the old one, `mv logs/<name>.log logs/<name>.<date>.log` first.

---

## 7. Git

### 7.1 What to commit

- Source under `src/`, `scripts/`, `tests/`.
- Job templates under `jobs/` (text, reproducible).
- YAML configs pinning experimental setups.
- Results **summaries** — `results/**/*.json`, `.csv`, `.md` tables.
  Small per-example predictions (<10 MB) OK.
- All the top-level `.md` docs.

### 7.2 What NOT to commit

- **Data.** No parallel corpora, no test sets, no tokenised shards.
- **Model weights or checkpoints.** `.gitignore` `*.safetensors`, `*.bin`,
  `*.pt`, `*.ckpt`.
- **Logs.** `logs/` is gitignored. PBS `.o*/.e*` files never enter git.
- **Caches.** Anything under `/scratch`, HF cache, torch, triton, vLLM.
- **Secrets.** No API keys, tokens, `.env` files. Read from env vars at
  runtime.
- **Manuscripts.** PDFs/LaTeX belong in a separate paper repo.

### 7.3 Attribution — hard rule

- **Never** add `Co-Authored-By: Claude`, `Generated with Claude Code`,
  or any AI-assistant trailer to commits, PRs, code comments, or any
  tracked file. This is a **double-blind submission rule** at ACL/EMNLP
  and IWSLT — an AI attribution in the public history is grounds for
  desk reject. There is no reason to add it earlier and rewrite later.
- Human author only. Do not add `Signed-off-by` unless a downstream
  requires it (nothing here does).

### 7.4 Commit style

- Imperative, present tense, prefix by area:
  - `annotator: batched prefix pass over top-k support`
  - `data: freeze WMT15 De→En newstest reference`
  - `results: llama3-8b OT tau=0.05 BLEU=28.1 AL=3.2 AL-CA=54ms/word`
  - `jobs: gpuhopper 4x SFT resume template`
  - `fix: correct monotonicity enforcement across chunk boundary`
- One logical change per commit.
- Body only if it explains *why*.

### 7.5 Cadence and branches

- Commit often; a working end-of-day state is a commit, not a stash.
- Push at least once per session. A commit on a compute node that gets
  rebuilt is a lost commit.
- `main` compiles and reflects current results. Feature work on
  short-lived branches. Never force-push `main`.

---

## 8. Logging and reproducibility

Every generated results JSON must record enough for someone else to
re-run it:

- `seed` (default 42, never `time.time()`)
- `model_name` and the exact `MODEL_BASE`-relative path used
- Framework versions — `torch`, `transformers`, `vllm`, `pot`
- For vLLM: version, `tensor_parallel_size`, GPU count. Greedy output is
  only deterministic within a fixed (version, TP, GPU-count) triple.
- Git commit hash (`git rev-parse HEAD`) of the running tree.
- The exact `tau` and `k` for the annotator; the exact
  `learning_rate`/`num_train_epochs`/`per_device_batch_size` for SFT.

Runners assemble this dict once at startup and dump it under a top-level
`"env"` key in every output JSON.

Also — **every run gets a `LOG.md` entry before the next one starts.**
The `LOG.md` template is already in the file. A run without an entry
did not happen.

---

## 9. Authorship (agreed at project start)

- **First author:** the student. Corresponding.
- **Last / supervising author:** Dipankar Srirag (UNSW).
- Order is locked. If a substantial technical contribution comes from
  someone else during the project (e.g., an alignment scorer we pull in),
  we add them in the middle — flag it in `LOG.md` at the time, not at
  submission.
- Target venue: ACL/EMNLP Findings or IWSLT. Both are double-blind at
  submission — see §7.3.

---

## 10. Things that will bite you (checklist)

- **Forgot `#PBS -l storage=...`** → job runs, `open()` on `/g/data/...`
  fails at first read. Log shows `PermissionError`.
- **Relative `-o` path** → log ends up in `$HOME`, not `logs/`. You'll
  think the job never ran.
- **Committed a `.log` or `.safetensors`** → `git filter-repo` to unroot;
  fix `.gitignore` before pushing.
- **Downloaded a model at runtime on a GPU queue** → no internet, wastes
  walltime. Always pre-download via `copyq`.
- **Overran walltime** → PBS kills at the boundary. No grace period.
  Checkpoint often; rely on the auto-resubmit contract.
- **vLLM output differs between two "identical" runs** → check `tp`,
  GPU count, and vLLM version. Greedy determinism is per-triple only.
- **Ran the annotator on the test set** → project invariant violated
  (`CLAUDE.md` §Non-negotiable invariants #1). Stop, log the incident,
  message me. Do not silently discard the outputs.
- **Committed with `Co-Authored-By: Claude`** → rewrite the commit
  before pushing. This must not enter the public history.
