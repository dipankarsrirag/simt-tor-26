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