# Experiments — ownership split

Who runs what. Configs live in `configs/`; plan and figure specs in `docs/followup-experiments.md`. Run each with `bin/run configs/{NN}.yaml --ngpus N`.

Model checkpoints are **never committed to git** (`.gitignore` covers `*.safetensors`, `results/train/**/final/`, etc.). Cross-check happens via **HuggingFace Hub** — every completed experiment gets pushed with its config + logs + eval JSONs + (where relevant) annotator matrices.

---

## Dipankar

**Focus:** all EAST-8B experiments (Fig 1, 2, 3 lines) + the shipped Gemma-2B baseline (Fig 4 left / already run).

| # | Config | What it is | Eval sets |
|---|---|---|---|
| 00 | `configs/00_gemma_2b_curated.yaml` | Gemma-2B + curated + self-annotate. **Already run** — artifacts in `_archive/results/gemma_2b_curated/`. Doubles as Fig 4 left. | WMT15 · WMT22 · IWSLT17 · IWSLT15 |
| 01 | `configs/01_east_8b_curated.yaml` | EAST-8B + curated + self-annotation. Fig 1, 2, 3, 4 right. | WMT15 · WMT22 · IWSLT17 · IWSLT15 |
| 02 | `configs/02_east_8b_machine_targets.yaml` | EAST-8B + curated sources + EAST-8B-machine-translated targets. Fig 1, 2 line 3 (Q1b — target-quality ablation). **Needs the retranslation preprocessing step first** (see config header). | WMT15 · WMT22 · IWSLT17 · IWSLT15 |
| 03 | `configs/03_east_8b_waitk.yaml` | EAST-8B + curated + **wait-k policy**. Fig 2, 3 line 4. **Needs `wait-k` criterion added to `scripts/07_annotate.py`.** | WMT22 · IWSLT17 · IWSLT15 |
| 04 | `configs/04_east_8b_conv.yaml` | EAST-8B + curated + **Conversational SiMT** (Wang 2024). Fig 2, 3 line 5. **Needs the conv-simt criterion + policy modules.** | WMT22 · IWSLT17 · IWSLT15 |

**First priority: push the existing Gemma-2B baseline to HF Hub** so Quang can pull its annotator matrices for 06/07:

```bash
bin/11_push_to_hub --config configs/00_gemma_2b_curated.yaml --org dipankarsrirag --private
```

That uploads the SFT checkpoint + tokenizer + `results/annotate/gemma-4-E2B-it/{pair}/matrices.jsonl` (the 8 matrices files) + config + logs + eval JSONs to `dipankarsrirag/simt-gemma_2b_curated`. Quang then pulls the annotator matrices via `bin/12_pull_from_hub` (below) before starting 06/07.

---

## Quang

**Focus:** scaling story (Fig 4 middle) + annotator portability story (Fig 5).

| # | Config | What it is | Depends on |
|---|---|---|---|
| 05 | `configs/05_gemma_4b_curated.yaml` | Gemma-4-E4B-it + curated + self-annotation. Fig 4 middle. | — |
| 06 | `configs/06_gemma_4b_from_2b_annot.yaml` | Gemma-4-E4B-it trained on **Gemma-2B's** annotations. Fig 5 middle. `--skip 2` (matrices come from Dipankar's HF push). | Dipankar's 00 must be pushed first. Pull matrices with `bin/12_pull_from_hub`. |
| 07 | `configs/07_east_8b_from_2b_annot.yaml` | EAST-8B trained on Gemma-2B's annotations. Fig 5 right. `--skip 2`. | Same as 06. |

**Pull Dipankar's Gemma-2B matrices before starting 06/07:**

```bash
bin/12_pull_from_hub --repo dipankarsrirag/simt-gemma_2b_curated
# → drops annotator matrices into results/annotate/gemma-4-E2B-it/{pair}/matrices.jsonl
# so 06/07's Stage 2 can be skipped.
```

**Deliverable per experiment (everyone):**
1. Populated `results/train/{tag}/final/` (SFT checkpoint — never committed to git).
2. Populated `results/eval/{tag}/*.json` (a JSON per test-set × direction × latency cell).
3. Populated `logs/{tag}/` — auto-written by `bin/run`. Contains `manifest.json` (git sha + hostname + GPUs + timestamps) and `stage_N_<name>.log` per stage.
4. **Push to HuggingFace Hub for verification:** `bin/11_push_to_hub --config configs/{tag}.yaml --org {your-hf-org} --private`. Uploads checkpoint + tokenizer + config + manifest + logs + eval JSONs + annotator matrices (when tag owns them) to `{your-org}/simt-{tag}`.
5. A `LOG.md` entry: config, command, headline numbers, any surprises.
6. Regenerate figures via `bin/run configs/{tag}.yaml --stage 6`.

---

## What everyone reads first

- **`README.md`** — repo tour, prerequisites, environment.
- **`configs/README.md`** — schema of a config YAML + shipped experiment table.
- **`docs/followup-experiments.md`** — the paper-facing experiment plan (figure specs, resolved design decisions Q1/Q2/Q3, risk register).
- **`docs/method.md`** — the annotator, mechanically.
- **`docs/data.md`** — corpus provenance, source URLs, download instructions.
- **`docs/setup.md`** — Gadi-specific paths + PBS notes (skip if not on Gadi; see `.simtrc.example` for the portable env-var setup).
- **`LOG.md`** — append-only decision + run log. **Read the tail** before starting anything.

---

## First-run checklist

Once, at the top of the project:

```bash
git clone https://github.com/dipankarsrirag/simt-tor-26.git
cd simt-tor-26
bash create-venv.sh
source .venv/bin/activate
cp .simtrc.example .simtrc
$EDITOR .simtrc                           # set SIMT_MODEL_BASE, SIMT_DATA_ROOT, SIMT_HF_CACHE
bin/01_download_training_data             # curated corpus (~12GB)
bin/02_download_wmt_test_sets             # WMT15/22 test sets
bin/03_download_iwslt_vi_test_set         # IWSLT15 vi-en test
bin/04_prepare_tokenizer --backbone {hf_id} --output results/train/{tag}/tokenizer
bin/05_probe_backbone --model_dir ${SIMT_MODEL_BASE}/{backbone}
huggingface-cli login                      # for bin/11_push_to_hub + bin/12_pull_from_hub
```

Then for each experiment:

```bash
bin/run configs/NN_{tag}.yaml --dry_run    # inspect commands
bin/run configs/NN_{tag}.yaml --ngpus 1     # go
# ...watch, log to LOG.md, iterate
bin/11_push_to_hub --config configs/NN_{tag}.yaml --org {your-hf-org} --private
```

Cross-annotation experiments (06, 07) — pull Dipankar's matrices first, then `--skip 2`:

```bash
bin/12_pull_from_hub --repo dipankarsrirag/simt-gemma_2b_curated
bin/run configs/06_gemma_4b_from_2b_annot.yaml --ngpus 1 --skip 2
```

Ping Dipankar when your 05 lands so figures can be aligned before firing 06/07.
